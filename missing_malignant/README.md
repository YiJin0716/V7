# Missed malignant nodules

The original V7 checkpoint (run 13220013, epoch 18) missed **18 malignant nodules in 9 test CTs**. Here malignant means the mean reader malignancy rating is greater than 3; these are reader ratings, not pathology outcomes.

Download an HTML file using **Raw / Download raw file**, then open it in a desktop browser. The files are self-contained and show the full CT, predicted and official masks, semantic features, and risk. Each CT is included once; use the nodule selector to locate the IDs below.

| CT / HTML | Missed malignant nodule IDs |
|---|---|
| [LIDC-IDRI-0259__scan259](LIDC-IDRI-0259__scan259.html) | 1 |
| [LIDC-IDRI-0326__scan327](LIDC-IDRI-0326__scan327.html) | 2 |
| [LIDC-IDRI-0404__scan408](LIDC-IDRI-0404__scan408.html) | 2, 4, 5, 7 |
| [LIDC-IDRI-0476__scan481](LIDC-IDRI-0476__scan481.html) | 3 |
| [LIDC-IDRI-0488__scan494](LIDC-IDRI-0488__scan494.html) | 2 |
| [LIDC-IDRI-0670__scan985](LIDC-IDRI-0670__scan985.html) | 1 |
| [LIDC-IDRI-0686__scan969](LIDC-IDRI-0686__scan969.html) | 10, 13, 14, 15, 17, 18, 19 |
| [LIDC-IDRI-0756__scan899](LIDC-IDRI-0756__scan899.html) | 1 |
| [LIDC-IDRI-0790__scan865](LIDC-IDRI-0790__scan865.html) | 2 |

The complete nodule list, reader mean malignancy, and diameter in mm are in [nodules.csv](nodules.csv). Other nodules in each CT remain visible for context.

Selection uses object probability ≥ 0.5, final mask probability ≥ 0.5, and one-to-one matching with best-reader mask IoU ≥ 0.1. The HTML viewer displays every individual reader mask and the official consensus mask. These nine selected CTs illustrate missed malignant nodules; they are not a representative performance sample.
