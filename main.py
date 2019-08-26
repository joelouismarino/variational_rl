from comet_ml import Experiment
import argparse
import torch
import numpy as np
from util.env_util import create_env
from agents import create_agent
from misc.buffer import Buffer
from misc.optimizer import Optimizer
from util.plot_util import Plotter
from util.train_util import train


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, help='environment name')
parser.add_argument('--device_id', default=None, type=int, help='GPU ID number')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
parser.add_argument('--train_seq_len', default=10, type=int, help='training sequence length')
parser.add_argument('--checkpoint_path', default=None, type=str, help='path to load saved weights')
parser.add_argument('--optimizer', default='adam', type=str, help='optimizer')
parser.add_argument('--grad_norm', default=None, help='gradient norm constraint')
parser.add_argument('--weight_decay', default=0., type=float, help='L2 weight decay')
# other arguments here
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device_id is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

# create the environment
env = create_env(args.env, args.seed)

# create the agent
agent, agent_args = create_agent(env, args.checkpoint_path, args.device_id)

# create the data buffer
buffer = Buffer(batch_size=args.batch_size, seq_len=args.train_seq_len)

# create the optimizer
optimizer = Optimizer(agent, optimizer=args.optimizer, lr=args.lr,
                      norm_grad=args.grad_norm, weight_decay=args.weight_decay)

# create the logger / plotter
plotter = Plotter(args, agent_args)

# train the agent
train(agent, env, buffer, optimizer, plotter)
