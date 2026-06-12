## Guide on setup data
- store data in `data/part1`
```
./data
└── part1
    ├── bacsidatnhkhoavitadoc_1.wav
    ├── bacsidatnhkhoavitadoc_2.wav
    ├── bacsidatnhkhoavitadoc_3.wav
    ├── bacsidatnhkhoavitadoc_4.wav
    ├── bacsinamnoitiet_1.wav
    ├── bacsinamnoitiet_2.wav
    ├── bacsingoanhtuandalieu_1.wav
    ├── bacsingoanhtuandalieu_2.wav
    ├── bacsingoanhtuandalieu_3.wav
    ├── bacsingoanhtuandalieu_4.wav
```

## Guide for running

1. Run on Modal.com
- in local, install modal package
```terminal
pip install modal
```
- setup modal account
```terminal
modal token set \
    --token-id  your_token_id \ 
    --token-secret your_token_secret
```

- run `modal_run.py` script
```terminal
modal run modal_run.py
```

2. Run with Docker
(update later)