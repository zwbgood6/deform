## Model Architecture

State Model 

<img src='./image/state_model.png' align="middle" width=360>

Action Model 

<img src='./image/action_model.png' align="middle" width=360>

Dynamics model 

<img src='./image/dynamics_model.png' align="middle" width=360>

## Hyperparameters

| Hyperparameters | Values | 
| :------------- | :----------: | 
|  epochs (overall) | 1000 | 
| epochs (state and action encoder-decoder) | 500 |
| epochs (dynamics model) | 500 |
| learning rate | 1e-3 |
| batch size | 32 |
| latent state size | 80 |
| latent action size | 80 |
| &lambda;<sub>1</sub> (action coefficient in the loss function) | 450 |
| &lambda;<sub>2</sub> (dynamics coefficient in the loss function) | 900 |
| &lambda;<sub>3</sub> (prediction coefficient in the loss function) | 10 |
