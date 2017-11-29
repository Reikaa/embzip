import argparse
import math
import os
import torch

from embzip.data import load_embeddings_txt, make_vocab, print_compression_stats, check_training
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
                        help='Number of embeddings to consider')
    parser.add_argument('--n_tables',
                        type=int,
                        required=True,
                        help='Number of embedding tables')
    parser.add_argument('--n_codes',
                        type=int,
                        required=True,
                        help='Number of embeddings in each table')
    parser.add_argument('--samples',
                        type=int,
                        default=200000,
                        help='Number of batches through the data')
    parser.add_argument('--report_every',
                        type=int,
                        default=1000,
                        help='Reporting interval in batches')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Number of embeddings in each batch')
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
                        help='Number of embeddings for validation')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='Run on GPU')
    args = parser.parse_args()

    # load embeddings
    train_g, valid_g = load_embeddings_txt(args.path, args.n_embs, args.valid)
    train_vocab = make_vocab(train_g)
    valid_vocab = make_vocab(valid_g) if valid_g else train_vocab

    # stats
    print_compression_stats(train_vocab, args.n_tables, args.n_codes)

    # model
    ec = EmbeddingCompressor(train_vocab['emb_size'], args.n_tables, args.n_codes, args.temp)

    # optimizer
    optim = torch.optim.Adam(ec.parameters(), lr=args.lr)

    # criterion
    criterion = torch.nn.MSELoss(size_average=False)

    losses = []
    old_params = None

    print('[CUDA]', args.cuda)
    if args.cuda:
        ec.cuda()
        criterion.cuda()

    samples = 0

    try:
        while samples < args.samples:

            # shuffle embeddings
            embeddings = train_vocab['embeddings'][torch.randperm(train_vocab['n_embs'])]

            # train
            for k in range(0, train_vocab['n_embs'], args.batch_size):
                samples += 1
                batch = embeddings[k:k+args.batch_size]

                y = ec(batch)

                loss = criterion(y, batch)
                losses.append(loss.data[0])

                #print(['%.4f' % x for x in y.data[0]])
                #print(['%.4f' % x for x in batch.data[0]])

                def euclidian_dist(x, y):
                    return math.sqrt(sum((i - j) ** 2 for i, j in zip(x, y)))
                #print(y.size(0))
                # for i in range(y.size(0)):
                #     print('E', euclidian_dist(y.data[i], batch.data[i]))
                #print('L', loss.data[0])
                #print()

                optim.zero_grad()
                loss.backward()
                optim.step()

                if samples % args.report_every == 0:
                    avg_train_loss = sum(losses) / len(losses) / args.batch_size
                    losses = []

                    # validate
                    v_batch = valid_vocab['embeddings']
                    if args.cuda:
                        v_batch = v_batch.cuda()
                    v_y = ec(v_batch)
                    v_loss = criterion(v_y, v_batch)
                    v_loss = v_loss.data[0] / v_batch.data.shape[0]
                    print('[%d] train: %.2f | valid: %.2f' % (samples, avg_train_loss, v_loss))

    except KeyboardInterrupt:
        # dump to file
        comp_emb_file = args.path + '.comp'
        print(comp_emb_file)
        with open(comp_emb_file, 'wt') as f:
            for word, emb in train_g.items():
                print(word)
                emb_comp = ec(emb).squeeze().cpu().data.numpy().tolist()
                for x in emb_comp:
                    word += ' ' + str(x)
                f.write(word + '\n')
