import argparse
import math
import os
import torch

from torch import nn
from time import time

from embzip.data import load_embeddings_txt, make_vocab, print_compression_stats, euclidian_dist, check_training, dump_reconstructed_embeddings, load_artichoke, save_artichoke
from embzip.model import EmbeddingCompressor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path',
                        required=True,
                        help='Path of embedding file')
    parser.add_argument('--n_embs',
                        type=int,
                        default=75000,
                        help='Number of fact_embs to consider')
    parser.add_argument('--n_tables',
                        type=int,
                        required=True,
                        help='Number of embedding tables')
    parser.add_argument('--n_codes',
                        type=int,
                        required=True,
                        help='Number of fact_embs in each table')
    parser.add_argument('--samples',
                        type=int,
                        default=int(2e7),
                        help='Number of embeddings to train')
    parser.add_argument('--report_every',
                        type=int,
                        default=1000,
                        help='Reporting interval in batches')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Number of fact_embs in each batch')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='Learning rate for optimizer')
    parser.add_argument('--temp',
                        type=float,
                        default=1,
                        help='Gumbel-Softmax temperature')
    parser.add_argument('--valid',
                        type=int,
                        default=1000,
                        help='Number of fact_embs for validation')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='Run on GPU')
    parser.add_argument('--hard',
                        action='store_true',
                        help='Use hard gumbel softmax')
    parser.add_argument('--load_artichoke',
                        action='store_true',
                        help='Load pre-trained artichoke embeddings')
    parser.add_argument('--save_artichoke',
                        action='store_true',
                        help='Save model into artichoke format')
    args = parser.parse_args()

    # load embeddings
    if args.load_artichoke:
        # from artichoke h5
        train_g, valid_g = load_artichoke(args.path)
    else:
        # from text file
        train_g, valid_g = load_embeddings_txt(args.path, args.n_embs, args.valid)

    train_vocab = make_vocab(train_g)
    valid_vocab = make_vocab(valid_g) if valid_g else make_vocab(train_g)

    # stats
    print_compression_stats(train_vocab, args.n_tables, args.n_codes)

    # model
    ec = EmbeddingCompressor(train_vocab['emb_size'], args.n_tables, args.n_codes, args.temp, args.hard)

    # optimizer
    optim = torch.optim.Adam(ec.parameters(), lr=args.lr)

    # criterion
    criterion = torch.nn.MSELoss(size_average=True)

    losses = []
    old_params = None

    print('[CUDA]', args.cuda)
    if args.cuda:
        ec.cuda()
        ec = nn.DataParallel(ec)
        criterion.cuda()

    samples = 0

    try:
        while samples < args.samples:

            # shuffle fact_embs
            embeddings = train_vocab['embeddings'][torch.randperm(train_vocab['n_embs'])]

            # train
            for k in range(0, train_vocab['n_embs'], args.batch_size):
                t0 = time()
                samples += args.batch_size
                batch = embeddings[k:k+args.batch_size]
                if args.cuda:
                    batch = batch.cuda()

                y_pred = ec(batch)

                loss = criterion(y_pred, batch)
                losses.append(loss.data[0])

                optim.zero_grad()
                loss.backward()
                optim.step()

                t1 = args.batch_size * 1000 * (time() - t0)

                if samples % args.report_every == 0:
                    # check validation set and report
                    avg_train_loss = sum(losses) / len(losses)
                    losses = []

                    # validate
                    v_batch = valid_vocab['embeddings']
                    if args.cuda:
                        v_batch = v_batch.cuda()

                    v_y = ec(v_batch)

                    v_loss = criterion(v_y, v_batch)
                    v_loss = v_loss.data[0]
                    avg_euc = sum([euclidian_dist(v_y.data[i], v_batch.data[i]) for i in range(v_y.size(0))]) / v_y.size(0)
                    print('[%d] train: %.5f | valid: %.5f | euc_dist: %.2f | %.2f emb/sec' % (samples, avg_train_loss, v_loss, avg_euc, t1))
                    #print([p.sum().data[0] for p in ec.parameters()])

    except KeyboardInterrupt:
        print('Training stopped!')

    # dump full size reconstructed embeddings to file
    #dump_reconstructed_embeddings(args.path + '.comp', ec, train_g)

    # save compressed embeddings in hdf5 format
    #:save_hdf5(args.path + '.h5', ec, train_g)
    if args.save_artichoke:
        save_artichoke(args.path + '.comp', ec, train_g)
