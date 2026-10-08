import argparse


def add_args():
    parser = argparse.ArgumentParser(description='GNN vulnerability detection with AVBCM')

    # ===== Data =====
    parser.add_argument('--data_path', type=str, default='datas/reveal',
                        help='dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='root directory for run outputs')

    # ===== Method switches =====
    parser.add_argument('--skip_init', action='store_true',
                        help='baseline run without parameter initialization')
    parser.add_argument('--use_avbc', action='store_true',
                        help='enable the AVBCM calibrator')
    parser.add_argument('--fixed_lambda', type=float, default=None,
                        help='freeze the calibration strength at a constant (ablation); '
                             'omit for the adaptive lambda_t = f(g_t)')
    parser.add_argument('--loss_function', default='ce',
                        choices=['ce', 'weighted_ce', 'focal', 'logit_adj', 'ldam', 'cb_focal'],
                        help='training loss')

    # ===== Model =====
    parser.add_argument('--model_name', type=str, default='gcn',
                        choices=['gcn', 'graphsage', 'gat', 'rgcn'],
                        help='GNN backbone')
    parser.add_argument('--in_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_relations', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)

    # ===== Training =====
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--seed', type=int, default=36, help='random seed')

    # ===== Device =====
    parser.add_argument('--device', type=str, default='cuda')

    return parser
