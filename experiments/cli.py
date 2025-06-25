import os
import sys
sys.path.append(".")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from openprompt.trainer import ClassificationRunner
from openprompt.pipeline_base import PromptForClassification, DomainDiscriminators, DomainKL
import argparse
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptDataLoader, PromptModel
from openprompt.prompts import load_template, load_verbalizer, load_template_generator, load_verbalizer_generator
from openprompt.utils.logging import config_experiment_dir, init_logger, logger
from openprompt.utils.calibrate import calibrate
from openprompt.config import get_yaml_config
from openprompt.plms import load_plm
from openprompt.data_utils import load_dataset
from openprompt.utils.cuda import model_to_device
from openprompt.utils.utils import check_config_conflicts

def get_config():
    parser = argparse.ArgumentParser("classification config")
    parser.add_argument("--config_yaml", type=str, help='the configuration file for this experiment.',
                        default="classification_nli_manual_prompt.yaml")
    parser.add_argument("--resume", action="store_true", help='whether to resume a training from the latest checkpoint.\
           It will fall back to run from initialization if no lastest checkpoint are found.', default=False)
    parser.add_argument("--test", action="store_true", help='whether to resume a training from the latest checkpoint.\
           It will fall back to run from initialization if no lastest checkpoint are found.',default=False) #

    args = parser.parse_args()
    config = get_yaml_config(args.config_yaml)
    check_config_conflicts(config)
    logger.info("CONFIGS:\n{}\n{}\n".format(config, "="*40))
    config.logging.path='../yaml'
    return config, args


def build_dataloader(dataset, template, tokenizer, config, split):
    dataloader = PromptDataLoader(dataset=dataset, 
                                template=template, 
                                tokenizer=tokenizer, 
                                batch_size=config[split].batch_size,
                                shuffle=True,
                                teacher_forcing=config[split].teacher_forcing \
                                    if hasattr(config[split],'teacher_forcing') else None,
                                predict_eos_token=True if config.task=="generation" else False,
                                **config.dataloader
                                )
    return dataloader

def save_config_to_yaml(config):
    from contextlib import redirect_stdout
    saved_yaml_path = os.path.join(config.logging.path, "config.yaml")
    with open(saved_yaml_path, 'w') as f:
        with redirect_stdout(f): print(config.dump())
    logger.info("Config saved as {}".format(saved_yaml_path))


def main():
    config, args = get_config()
    print(config.dataset.target_domain)
    if not args.resume:
        save_config_to_yaml(config)
    # set seed
    # torch.manual_seed(321)
    set_seed(config)
    # load the pretrained models, its model, tokenizer, and config.
    plm_model, plm_tokenizer, plm_config = load_plm(config)
    # load dataset. The valid_dataset can be None
    train_dataset, valid_dataset, test_dataset, Processor = load_dataset(config)
    # for set in train_dataset:
    #     print(len(set))
    # exit(0)

    # define prompt
    template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)


    # template = model_to_device(template, config.environment)
    verbalizer = load_verbalizer(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config, classes=Processor.labels)
    # verbalizer = model_to_device(verbalizer, config.environment)

    # load prompt’s pipeline model
    prompt_model = PromptForClassification(plm_model, template, verbalizer) # 预训练模型用于分类预测
    domain_adv = DomainDiscriminators(config) # 用于领域对齐
    domain_kl = DomainKL(config) # 用于词汇表分布对齐


    # move the model to device:
    prompt_model = model_to_device(prompt_model, config.environment)
    domain_adv = model_to_device(domain_adv, config.environment)
    domain_kl = model_to_device(domain_kl, config.environment)
    # process data and get data_loader
    
    if config.calibrate is not None:
        assert isinstance(prompt_model, PromptForClassification), "The type of model doesn't support calibration."
        calibrate(prompt_model, config)


    # train_dataloader = build_dataloader(train_dataset, template, plm_tokenizer, config, "train")
    if not config.dataset.domains:
        train_dataloader = build_dataloader(train_dataset, template, plm_tokenizer, config, "train")
    else:
        train_dataloader = [build_dataloader(dataset, template, plm_tokenizer, config, "train") for dataset in train_dataset]

    # train_dataloader = build_dataloader(train_dataset, template, plm_tokenizer, config, "train")

    if valid_dataset is None:
        valid_dataset = test_dataset
    # tokenizing
    valid_dataloader = build_dataloader(valid_dataset, template, plm_tokenizer, config, "dev")
    test_dataloader = build_dataloader(test_dataset, template, plm_tokenizer, config, "test")

    if config.task == "classification":
        runner = ClassificationRunner(prompt_model = prompt_model,
                                domain_adv = domain_adv,
                                domain_kl = domain_kl,
                                train_dataloader = train_dataloader,
                                valid_dataloader = valid_dataloader,
                                test_dataloader = test_dataloader,
                                config = config)
        
    else:
        raise NotImplementedError
    if not args.resume:
        runner.run()
    else:
        if args.test: #
            runner.test()#
        # else:#
        #     runner.resume()




if __name__ == "__main__":
    main()
    # get_config()
