## Dependency

```
python 3.6
pytorch >= 1.4.0
tqdm
```

## Usage

```python
from attack import PGD, FGSM

attack_config = {
    'eps' : 8.0/255.0, 
    'attack_steps': 7,
    'attack_lr': 2.0 / 255.0, 
    'random_init': False, 
}

pgd = PGD(model, attack_config)
adversarial_image = pgd(image, label)

attack_config = {
    'eps' : 8.0/255.0,
    'random_init' : False,
}

fgsm = FGSM(model, attack_config)
adversarial_image = fgsm(image, label)
```

## Reference

- Jeffkang-94 [pytorch-adversarial-attack](https://github.com/Jeffkang-94/pytorch-adversarial-attack)