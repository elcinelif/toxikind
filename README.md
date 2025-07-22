# ToxiKind
## Quick description
**AI-Based Toxicity Prediction to Minimize Harm\
(for both 🐭🐰🐹 &🧍)**

This is a final project of the [*Data Science & AI* course @ Le Wagon](https://www.lewagon.com/data-science-course), presented 2025-06-20.

## Links
- [Project Presentation](https://www.youtube.com/watch?v=CjB8OIFrwjY)
- [Streamlit Interface](https://toxikind.streamlit.app/)
- [GitHub Backend (**this repository**)](https://github.com/elcinelif/toxikind)
- [GitHub Frontend](https://github.com/elcinelif/toxikind-frontend)

# Goal
Animal testing for drug discovery has serious limitations:
- 90% of drugs that pass animal tests fail in human trials.
- It takes about $ 3 milion and 10 years to develop a *single* compound.
- Each year, milions of animals are being tested.

Machine learning opens up many opportunities to reduce costs and the number of test animals:
- Trained on human-relevant experimental data.
- Fast *in-silico* screening of thousands of compounds.
- Reduces preclinical animal testing & harm.

Goals of this project are:
- to predict the toxicity (to humans) of chemical compounds as reliably as possible.
- to provide an [Interface](https://toxikind.streamlit.app/) for experts and the interested public. It includes numeric predictions and both graphic and verbal desriptions of the toxicity. The latter using an large-language model.

The performance is measured using the [F1-Score](https://en.wikipedia.org/wiki/F-score), a harmonic mean balancing both the precision and recall of a supervised predictive model.

# Data
This project uses the [*Tox21*](http://bioinf.jku.at/research/DeepTox/tox21.html) dataset with more than 8.000 compounds, each having about 800 chemical properties. The dependent variables are toxicities for twelve biological assays in binary form.

# Model
In the current project status, the [*Gradient Boosting Classifier*](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) from Scikit-Learn is being used.

The model hyperparameters are:
- Number of estimators = 300
- Learning rate = 0.1
- Maximum depth = 4

# State & Future of the Project
## Performance
Currently, six out of twelve assays perform good enough for being implemented in the frontend.

| Biological assay                  | Why It Matters?                                                      | F1-Score |
|----------------------------------:|:---------------------------------------------------------------------|:---------|
| Aryl Hydrocarbon (NR-AhR)         | May signal cancer or immune effects                                  | 0.63     |
| Androgen Receptor (NR-AR)         | Can affect development or reproduction                               | 0.59     |
| Androgen Receptor LBD (NR-AR-LBD) | Refines hormone disruption assessment                                | 0.64     |
| Estrogen Receptor LBD (NR-ER-LBD) | Indicates hormone disruption risk                                    | 0.51     |
| Antioxidant Response (SR-ARE)     | Shows potential for cell damage                                      | 0.52     |
| Mitochondrial Membrane (SR-MMP)   | Disrupts cell’s energy production, can lead to cell stress and death | 0.65     |

The F1-Scores of the remaining assays are too low so they cannot be taken into account.

## Postprocessing remaining
- Refactoring remaining notebook code to Python files.
- Proper *main.py* and *Makefile*.
- Documention in the interface.
- General cleanup.

## Future Ideas
Existing project structure:
- Collect further data to overcome data imbalance & scarcity.
- Try more machine learning models to improve F1-Score, so all assays can be predicted.

Possible restructuring of the project:
- Enable prediction of any compound by use of modern featurization tools to extract chemical properties.

# Contributors
- [Elif Elçin](https://github.com/elcinelif) (Owner & Project Lead)
- [Abdul Bakr](https://github.com/madpythonista)
- [Bart Dutkiewicz](https://github.com/bartdutkiewicz)
- [Ivan Kostov](https://github.com/kostovI)
