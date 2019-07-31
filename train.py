import os
import argparse
import logging
import json

from model import get_model
from tokenizer import VNTokenizer
from data_loader import load_tokenizer, build_data_iter
from utils import train_model, evaluate_model, forward_and_loss

import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tokenizer_path')
    parser.add_argument('train_path')
    parser.add_argument('val_path')
    parser.add_argument('--config_file', default='model_config.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--weight_dir', default='weight')
    parser.add_argument('--restore_file', default=None)
    parser.add_argument('--log_file', default='log.txt')

    args = parser.parse_args()

    return args


if __name__=='__main__':
    args = get_args()

    # Init logging
    print("Initializing logger...")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file,
                        filemode='a',
                        level=logging.INFO,
                        format="%(levelname)s - %(asctime)s: %(message)s")
    logger=logging.getLogger(__name__)

    # Load tokennizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    pad_token = tokenizer.w2i[tokenizer.pad_token]

    # Preparing data
    print("Preparing data...")
    train_iter = build_data_iter(args.train_path, tokenizer, batch_size=args.batch_size)
    val_iter = build_data_iter(args.val_path, tokenizer, batch_size=args.batch_size)

    # Load model config
    vocab_size = len(tokenizer.w2i)
    if not os.path.isfile(args.config_file):
        raise Exception("Invalid config file")
    with open(args.config_file) as f:
        model_config = json.load(f)
    model_config['vocab_size'] = vocab_size

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Init model
    print("Initializing model...")
    model = get_model(**model_config)
    if device.type=='cuda':
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    print("Using %s" % device.type)

    # Load weight
    if args.restore_file is not None:
        if not os.path.isfile(args.restore_file):
            raise Exception("Invalid weight paht")

        print("Load model")
        state = torch.load(args.restore_file)
        model.load_state_dict(state['model'])
        optim.load_state_dict(state['optim'])

    # Init weight dir
    if not os.path.isdir(args.weight_dir):
        os.makedirs(args.weight_dir)

    # Train model
    print("Start training %d epochs" % args.num_epochs)
    for e in range(1, args.num_epochs+1):
        logger.info("Epoch %02d/%02d" % (e, args.num_epochs))
        logger.info("Start training")

        print("\nEpoch %02d/%02d" % (e, args.num_epochs), flush=True)
        
        save_file = None
        if e % args.save_every == 0:
            save_file = os.path.join(args.weight_dir, 'epoch_%02d.h5' % e)

        train_loss = train_model(model, optim, train_iter, pad_token, device=device, weight_path=save_file)
        logger.info("End training")
        logger.info("train_loss = %.8f" % train_loss)
        val_loss = evaluate_model(model, val_iter, pad_token, device=device)
        logger.info("val_loss   = %.8f\n" % val_loss)

