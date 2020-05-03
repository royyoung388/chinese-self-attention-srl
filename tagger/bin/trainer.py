# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import logging
import os
import re
import socket
import time

import matplotlib.pyplot as plt
import six
import torch
import torch.distributed as dist

import tagger.data as data
import tagger.models as models
import tagger.optimizers as optimizers
import tagger.utils as utils
import tagger.utils.summary as summary
from tagger.utils.validation import ValidationWorker
from tagger.utils.validationThread import ValidationWorkerThread


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training SRL tagger",
        usage="trainer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str,
                        help="Path of the training corpus")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to saved models")
    parser.add_argument("--vocabulary", type=str, nargs=2,
                        help="Path of source and target vocabulary")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training mode")
    parser.add_argument("--local_rank", type=int,
                        help="Local rank of this process")
    parser.add_argument("--half", action="store_true",
                        help="Enable mixed precision training")
    parser.add_argument("--hparam_set", type=str,
                        help="Name of pre-defined hyper parameter set")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    parser.add_argument("--subthread", action="store_true",
                        help="use sub thread to run validation")
    return parser.parse_args(args)


def default_params():
    params = utils.HParams(
        input="",
        output="",
        model="transformer",
        vocab=["", ""],
        pad="<pad>",
        bos="<eos>",
        eos="<eos>",
        unk="<unk>",
        # Dataset
        batch_size=4096,
        fixed_batch_size=False,
        min_length=1,
        max_length=256,
        buffer_size=1024,
        # Initialization
        initializer_gain=1.0,
        initializer="uniform_unit_scaling",
        # Regularization
        scale_l1=0.0,
        scale_l2=0.0,
        # Training
        script="",
        warmup_steps=4000,
        train_steps=100000,
        update_cycle=1,
        optimizer="Adam",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        adadelta_rho=0.95,
        adadelta_epsilon=1e-6,
        clipping="global_norm",
        clip_grad_norm=5.0,
        learning_rate=1.0,
        learning_rate_schedule="linear_warmup_rsqrt_decay",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        device_list=[0],
        embedding="",
        # Plot
        plot_step=50,
        # Validation
        keep_top_k=50,
        frequency=10,
        # Checkpoint Saving
        early_stopping=10,
        keep_checkpoint_max=20,
        keep_top_checkpoint_max=5,
        save_summary=True,
        save_checkpoint_secs=0,
        save_checkpoint_steps=500,
    )

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if not os.path.exists(p_name) or not os.path.exists(m_name):
        return params

    with open(p_name) as fd:
        logging.info("Restoring hyper parameters from %s" % p_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    with open(m_name) as fd:
        logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def export_params(output_dir, name, params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)

    with open(filename, "w") as fd:
        fd.write(params.to_json())


def merge_params(params1, params2):
    params = utils.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def override_params(params, args):
    params.model = args.model or params.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.vocab = args.vocabulary or params.vocab
    params.parse(args.parameters)

    src_vocab, src_w2idx, src_idx2w = data.load_vocabulary(params.vocab[0])
    tgt_vocab, tgt_w2idx, tgt_idx2w = data.load_vocabulary(params.vocab[1])

    params.vocabulary = {
        "source": src_vocab, "target": tgt_vocab
    }
    params.lookup = {
        "source": src_w2idx, "target": tgt_w2idx
    }
    params.mapping = {
        "source": src_idx2w, "target": tgt_idx2w
    }

    return params


def collect_params(all_params, params):
    collected = utils.HParams()

    for k in six.iterkeys(params.values()):
        collected.add_hparam(k, getattr(all_params, k))

    return collected


def print_variables(model):
    weights = {v[0]: v[1] for v in model.named_parameters()}
    total_size = 0

    for name in sorted(list(weights)):
        v = weights[name]
        print("%s %s" % (name.ljust(60), str(list(v.shape)).rjust(15)))
        total_size += v.nelement()

    print("Total trainable variables size: %d" % total_size)


def save_checkpoint(step, epoch, model, optimizer, params, loss_record):
    if dist.get_rank() == 0:
        state = {
            "step": step,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss_record,
        }
        utils.save(state, params.output, params.keep_checkpoint_max)


def infer_gpu_num(param_str):
    result = re.match(r".*device_list=\[(.*?)\].*", param_str)

    if not result:
        return 1
    else:
        dev_str = result.groups()[-1]
        return len(dev_str.split(","))


def get_clipper(params):
    if params.clipping.lower() == "none":
        clipper = None
    elif params.clipping.lower() == "adaptive":
        clipper = optimizers.adaptive_clipper(0.95)
    elif params.clipping.lower() == "global_norm":
        clipper = optimizers.global_norm_clipper(params.clip_grad_norm)
    else:
        raise ValueError("Unknown clipper %s" % params.clipping)

    return clipper


def get_learning_rate_schedule(params):
    if params.learning_rate_schedule == "linear_warmup_rsqrt_decay":
        schedule = optimizers.LinearWarmupRsqrtDecay(params.learning_rate,
                                                     params.warmup_steps)
    elif params.learning_rate_schedule == "piecewise_constant_decay":
        schedule = optimizers.PiecewiseConstantDecay(
            params.learning_rate_boundaries, params.learning_rate_values)
    elif params.learning_rate_schedule == "linear_exponential_decay":
        schedule = optimizers.LinearExponentialDecay(params.learning_rate,
                                                     params.warmup_steps, params.start_decay_step,
                                                     params.end_decay_step,
                                                     dist.get_world_size())
    else:
        raise ValueError("Unknown schedule %s" % params.learning_rate_schedule)

    return schedule


def broadcast(model):
    for var in model.parameters():
        dist.broadcast(var.data, 0)


def main(args):
    model_cls = models.get_model(args.model)

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = default_params()
    params = merge_params(params, model_cls.default_params(args.hparam_set))
    params = import_params(args.output, args.model, params)
    params = override_params(params, args)

    # Initialize plot
    loss_record = [[], []]
    plt.xlabel('step')
    plt.ylabel('loss')

    # Initialize distributed utility
    if args.distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(args.local_rank)
    else:
        dist.init_process_group("nccl", init_method=args.url,
                                rank=args.local_rank,
                                world_size=len(params.device_list))
        torch.cuda.set_device(params.device_list[args.local_rank])
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Export parameters
    if dist.get_rank() == 0:
        export_params(params.output, "params.json", params)
        export_params(params.output, "%s.json" % params.model,
                      collect_params(params, model_cls.default_params()))

    model = model_cls(params).cuda()
    model.load_embedding(params.embedding)

    if args.half:
        model = model.half()
        torch.set_default_dtype(torch.half)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model.train()

    # Init tensorboard
    summary.init(params.output, params.save_summary)
    schedule = get_learning_rate_schedule(params)
    clipper = get_clipper(params)

    if params.optimizer.lower() == "adam":
        optimizer = optimizers.AdamOptimizer(learning_rate=schedule,
                                             beta_1=params.adam_beta1,
                                             beta_2=params.adam_beta2,
                                             epsilon=params.adam_epsilon,
                                             clipper=clipper)
    elif params.optimizer.lower() == "adadelta":
        optimizer = optimizers.AdadeltaOptimizer(
            learning_rate=schedule, rho=params.adadelta_rho,
            epsilon=params.adadelta_epsilon, clipper=clipper)
    else:
        raise ValueError("Unknown optimizer %s" % params.optimizer)

    if args.half:
        optimizer = optimizers.LossScalingOptimizer(optimizer)

    optimizer = optimizers.MultiStepOptimizer(optimizer, params.update_cycle)

    if dist.get_rank() == 0:
        print_variables(model)

    dataset = data.get_dataset(params.input, "train", params)

    # Load checkpoint
    checkpoint = utils.latest_checkpoint(params.output)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        step = state["step"]
        epoch = state["epoch"]
        model.load_state_dict(state["model"])

        if "loss" in state:
            loss_record = state["loss"]
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
    else:
        step = 0
        epoch = 0
        broadcast(model)

    def train_fn(inputs):
        features, labels = inputs
        loss = model(features, labels)
        return loss

    counter = 0
    should_save = False

    if params.script:
        if params.subthread:
            thread = ValidationWorkerThread(daemon=True)
            thread.init(params)
            thread.start()
        else:
            valiation = ValidationWorker(params)
    else:
        valiation = None

    def step_fn(features, step):
        t = time.time()
        features = data.lookup(features, "train", params)
        loss = train_fn(features)
        gradients = optimizer.compute_gradients(loss,
                                                list(model.parameters()))
        if params.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           params.clip_grad_norm)

        optimizer.apply_gradients(zip(gradients,
                                      list(model.named_parameters())))

        t = time.time() - t

        summary.scalar("loss", loss, step, write_every_n_steps=1)
        summary.scalar("global_step/sec", t, step)

        if step % params.plot_step == 0:
            loss_record[0].append(step)
            loss_record[1].append(float(loss))

        print("epoch = %d, step = %d, loss = %.3f (%.3f sec)" %
              (epoch + 1, step, float(loss), t))

    try:
        while True:
            for features in dataset:
                # if valiation.is_early_stopping():
                #     return

                if counter % params.update_cycle == 0:
                    step += 1
                    utils.set_global_step(step)
                    should_save = True

                counter += 1
                step_fn(features, step)
                del features
                # for f in features:
                #     if type(f) == dict:
                #         for i in f.values():
                #             i.dispose()
                #     else:
                #         f.dispose()

                if step % params.save_checkpoint_steps == 0:
                    if should_save:
                        plt.plot(loss_record[0], loss_record[1])
                        plt.savefig(os.path.join(params.output, "loss.png"))
                        save_checkpoint(step, epoch, model, optimizer, params, loss_record)
                        valiation.val()
                        should_save = False

                if step >= params.train_steps:
                    if should_save:
                        plt.plot(loss_record[0], loss_record[1])
                        plt.savefig(os.path.join(params.output, "loss.png"))
                        save_checkpoint(step, epoch, model, optimizer, params, loss_record)
                        valiation.val()

                    if dist.get_rank() == 0:
                        summary.close()

                    return

            epoch += 1
    finally:
        if valiation is not None:
            valiation.stop()


# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)


if __name__ == "__main__":
    parsed_args = parse_args()

    if parsed_args.distributed:
        main(parsed_args)
    else:
        # Pick a free port
        with socket.socket() as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
            url = "tcp://localhost:" + str(port)
            parsed_args.url = url

        world_size = infer_gpu_num(parsed_args.parameters)

        if world_size > 1:
            torch.multiprocessing.spawn(process_fn, args=(parsed_args,),
                                        nprocs=world_size)
        else:
            process_fn(0, parsed_args)
