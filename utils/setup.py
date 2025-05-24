from torch.optim import lr_scheduler

def get_scheduler(params, optimizer):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + params.epoch_count - params.n_epochs) / float(params.n_epochs_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler
    