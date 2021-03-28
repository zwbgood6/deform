## Model Architecture


## Hyperparameters

| Hyperparameters | Values | 
| :------------- | :----------: | 
|  epochs (Overall) | 1000 | 
| epochs (state and action encoder-decoder) | 500 |
| epochs (dynamics model) | 500 |
| learning rate | 1e-3 |
| batch size | 32 |
| latent state size | 80 |
| latent action size | 80 |
| &lambda<sub>1 (action coefficient in the loss function) | 450 |
| $\lambda_2$ (dynamics coefficient in the loss function) | 900 |
| $$\lambda_3$$ (predictio coefficient in the loss function) | 10 |
