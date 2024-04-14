#%%
import wandb
api = wandb.Api()
api = wandb.Api()

sweep = api.sweep("aizu-nerf/nerf-qa-2/h3luv013")
sweep.config
#%%
sweep.config['parameters'][]

#%%
sweep.update()