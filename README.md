# isup-grading

## Project Structure
```
├── conf                # Configuration files
├── data                # Data files
│   ├── external        # External data
│   ├── interim         # Intermediate processed data
│   ├── processed       # Processed data
│   └── raw             # Raw data (read only)
├── isupgrader          # Main project source code
│   ├── data            # Code that handles data processing
│   ├── executors       # Scripts that run code end-to-end (though refer notebooks)
│   ├── models
│   ├── networks        # Interchangeable networks that the models rely on
│   └── utils           # Utils than can be used in several places
├── local               # A non-pushed dir for local work
├── models              # Compiled models
├── notebooks           # Notebooks used for literate programming and executors
├── references          # Manuals and other material
├── reports
└── results             # Results from training

```

## Conda
See `Makefile` for creating, updating, and cleaning.

### Activating
`conda activate ./env`