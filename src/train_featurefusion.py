import models,utils
from training_setup import TrainingSetup, cli
import torch
from tqdm import tqdm
import torch.utils.data

def load_dataset():
    print('Loading dataset...')

    return utils.Spec2MolDataset(
            'nist20_and_gnps_pos_00000000000_f',
            '/data/zhangxiaofeng/code/code/data/compound_based_split/train.tsv',
            parser=utils.gnps_parser,
            mol_representation=utils.hunhe_fingerprint1,
            use_cache = True,
            )

# def load_dataset():
#     print('Loading dataset...')

#     return utils.Spec2MolDataset(
#             'nist20_and_gnps_pos',
#             '/data/zhangxiaofeng/code/code/data/data/data_processed/train_data.tsv',
#             parser=utils.enhanced_gnps_parser,
#             mol_representation=utils.hunhe_fingerprint1,
#             use_cache = True,
#             )

# def load_models(hparams):
#     print('Loading models...')

#     input_dim = utils.ENHANCED_SPECTRA_DIM

#     _models = {}
#     for hdim in hparams['hdim']:
#         for n_layers in hparams['n_layers']:
#             _models[f'enhanced_featurefusion_{hdim}_layers_{n_layers}'] = models.AdvancedSpec2MolModel(
#                     spectrum_dim=input_dim,
#                     prop_dim=utils.FINGERPRINT_NBITS+857+167+ len(utils.FRAGMENT_LEVELS) + len(utils.ADDUCTS) ,
#                     hdim=hdim,
#                     n_layers=n_layers
#                     )
#     return _models

def load_models(hparams):
    print('Loading models...')
    _models = {}
    for hdim in hparams['hdim']:
        for n_layers in hparams['n_layers']:
            _models[f'featurefusion_hunhe_nist_{hdim}_layers_{n_layers}_new'] = models.Spec2MolFeaturefusion(
                    molecule_dim=utils.SPECTRA_DIM ,
                    prop_dim=utils.FINGERPRINT_NBITS+857+167+ len(utils.FRAGMENT_LEVELS) + len(utils.ADDUCTS) ,
                    hdim=hdim,
                    n_layers=n_layers,

                    )
    return _models
SCAN_HPARAMS = {
    'hdim': [512, 1024, 2048],
    'n_layers': [1, 2, 3, 5, 6, 7],
    'batch_size': [16, 32, 64, 128, 256, 512]
}
# SCAN_HPARAMS = {
#     'hdim': [1024],
#     'n_layers': [4],
#     'batch_size': [16]
# }
PROD_HPARAMS = {
    'hdim': [2048],
    'n_layers': [6],
    'batch_size': [512]
}

def main():
    setup_args, clargs, hparams = cli(SCAN_HPARAMS, PROD_HPARAMS)

    dataset = load_dataset()

    _models = load_models(hparams)

    setup_args['n_epochs'] = min(500, setup_args['n_epochs'])


    setups = {}
    for bsz in hparams['batch_size']:
        for mname, model in _models.items():
            suffix = '_prod' if clargs.prod else ''
            setup_name = f'model_{mname}_bs_{bsz}_adam{suffix}'
            setups[setup_name] = TrainingSetup(
                    model=model,
                    dataset=dataset,
                    outdir=f'runs_pos_00000000000_f/{setup_name}',
                    batch_size=bsz,
                    dataloader=torch.utils.data.DataLoader,
                    **setup_args
                    )
    print(list(setups.items()))
    pbar = tqdm(list(setups.items()))
    for name, tsetup in pbar:
        pbar.set_description(f'Tng: {name}')
        tsetup.train()


if __name__ == '__main__':
    main()

