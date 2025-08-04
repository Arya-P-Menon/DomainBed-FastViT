### Replicated - Multi-domain generalisation

for command in delete_incomplete launch
do
  python -m domainbed.scripts.sweep ${command} --data_dir=./data \
  --output_dir=./output_cvt13 --command_launcher "multi_gpu" --algorithms ERM_SPSDViT \
  --single_test_envs  --datasets DR --n_hparams 1 --n_trials 3  \
  --hparams """{\"backbone\":\"CVTTiny\",\"batch_size\":16,\"lr\":5e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0}"""
done

### Replicated - Single-domain generalisation

for command in delete_incomplete launch
do
  python -m domainbed.scripts.sweep_updated ${command} --data_dir=./data \
  --output_dir=./output_SDG_cvt13 --command_launcher "multi_gpu" --algorithms ERM_SPSDViT \
  --single_domain_generalization  --datasets DR --n_hparams 1 --n_trials 3  \
  --hparams """{\"backbone\":\"CVTSmall\",\"batch_size\":16,\"lr\":5e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0}"""
done

###############################################################################################################################

### FastVit - Multi-domain generalisation

for command in delete_incomplete launch
do
  python -m domainbed.scripts.sweep ${command} --data_dir=./data \
  --output_dir=./output_fastViT/ --command_launcher "multi_gpu" --algorithms FastViT \
  --single_test_envs  --datasets DR --n_hparams 1 --n_trials 3  \
  --hparams """{\"backbone\":\"fastViT\",\"batch_size\":16,\"lr\":5e-5,\"resnet_dropout\":0.0,\"weight_decay\":0.00}"""
done

### FastVit - Single-domain generalisation

for command in delete_incomplete launch
do
  python -m domainbed.scripts.sweep_updated ${command} --data_dir=./data \
  --output_dir=./output_SDG_fastViT/ --command_launcher "multi_gpu" --algorithms FastViT \
  --single_domain_generalization  --datasets DR --n_hparams 1 --n_trials 3  \
  --hparams """{\"backbone\":\"fastViT\",\"batch_size\":16,\"lr\":5e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0}"""
done

#####################################################################################


