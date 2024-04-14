#%%
import wandb
api = wandb.Api()
api = wandb.Api()

sweep = api.sweep("aizu-nerf/nerf-qa-2/h3luv013")
sweep.config
#%%
sweep.config['parameters']['entropy_loss_coeff'] = {
    'distribution': 'q_log_uniform_values',
    'max': 1.0,
    'min': 0.0,
    'q': '1e-06'
}
#%%
sweep.config

#%%
sweep.update()