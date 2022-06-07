## Dependency

```
python 3.6
pytorch >= 1.4.0
tqdm
```

## Usage

```python
sys.path.append("..")
from attack import pgd, fgsm

attack_config = {
    'eps' : 8.0/255.0, 
    'attack_steps': 7,
    'attack_lr': 2.0 / 255.0, 
    'random_init': False, 
}

pgd = pgd.PGD(model, attack_config)
# image of size [N,H,W], label of size [N]
adversarial_image = pgd(image.squeeze(dim=0), label.squeeze(dim=0)) 

attack_config = {
    'eps' : 8.0/255.0,
    'random_init' : False,
}

fgsm = fgsm.FGSM(model, attack_config)
adversarial_image = fgsm(image.squeeze(dim=0), label.squeeze(dim=0))
```

## Reference

- Jeffkang-94 [pytorch-adversarial-attack](https://github.com/Jeffkang-94/pytorch-adversarial-attack)