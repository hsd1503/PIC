# PIC

## Usage

`python exp.py`

The reproducible results are saved in `run_exp.ipynb`. 

You can also run `prepare_firsthours.py` to extract `icu_first24hours.csv` from raw csv files. 

## Dataset

PIC (paediatric Intensive Care) dataset. A large, single-center database comprising information relating to patients admitted to critical care units at the Children's Hospital of Zhejiang University School of Medicine. http://pic.nbscn.org/

* Xian Zeng#, Gang Yu#, Yang Lu#,Linhua Tan, Xiujing Wu, Shanshan Shi, Huilong Duan, Qiang Shu* and Haomin Li*. PIC, a paediatric-specific intensive care database. Scientific Data 2020 7:14 DOI:doi.org/10.1038/s41597-020-0355-4. Available from: https://www.nature.com/articles/s41597-020-0355-4

## Tasks

Predicting In-hospital Mortality of Patients in the Paediatric ICU

* Pollack, Murray M., Kantilal M. Patel, and Urs E. Ruttimann. "PRISM III: an updated Pediatric Risk of Mortality score." Critical care medicine 24.5 (1996): 743-752.

## (TODO) Extract AKI patients
* Plötz, Frans B., et al. "Pediatric acute kidney injury in the ICU: an independent evaluation of pRIFLE criteria." Intensive care medicine 34.9 (2008): 1713-1717.

AKI is defined as any of the following (Not Graded):
- Increase in SCr by >=0.3 mg/dl (>=26.5 lmol/l) within 48 hours; or
- Increase in SCr to >=1.5 times baseline, which is known or presumed to have occurred within the prior 7 days; or 
- Urine volume <0.5 ml/kg/h for 6 hours.
