Code structure:

```
zero_order_optim/
├── __init__.py
├── base.py
├── perturbations.py
├── optimizers/
│   ├── __init__.py
│   ├── opt_utils/
│   ├── zo_sgd.py
│   ├── zo_adam.py
│   ├── zo_conservative.py
│   ├── zo_sign_sgd.py
│   ├── forward_grad.py
│   ├── jaguar_sign_sgd.py
│   ├── jaguar_muon.py
│   └── zo_muon.py
└── utils.py
```

Добавление собственного оптимизатора 

Для того, чтобы реализовать собственный оптимизатор, необходимо написать класс для него в папке `optimizers`, отнаследовав его от `ZeroOrderOptimizer`. 

Далее добавить его импорт в файле `__init__.py` той же папки.

Потом добавить его в обработку в `trainer.py` (`if args.trainer ==`).

При запуске необходимо указать соотвествующий аргумент `--trainer` для метода в скрипте `run_script.sh`. s

Пертрубация параметров производится с помощью функции `matrix_pertrub_params`.

