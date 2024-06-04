# UA-BCD
The official implement of 《Overcome the Uncertainty Challenges in Detecting Building Changes from Remote Sensing Images》
## Performance Comparison with Baseline Models on the LEVIR-CD Dataset

$\uparrow$ indicates the higher score the better and vice versa. The best score for each metric is marked in **bold**. The second score for each metric is _underlined_.

| Baseline       | IoU $\uparrow$ | F1 $\uparrow$ | Pre $\uparrow$ | Recall $\uparrow$ |
|----------------|----------------|---------------|----------------|-------------------|
| FC-Siam-conc   | 82.24          | 90.25         | 89.67          | 90.84             |
| FC-Siam-diff   | 82.24          | 90.26         | 90.40          | 90.11             |
| SNUNet         | 82.65          | 90.50         | 90.46          | 90.54             |
| BIT            | 82.76          | 90.56         | 90.87          | 90.26             |
| ChangeFormer   | 81.25          | 89.65         | 89.70          | 89.61             |
| P2V-CD         | 83.67          | 91.11         | 91.01          | _91.21_           |
| HANet          | 82.27          | 90.28         | 91.21          | 89.36             |
| CGNet          | _85.21_        | _92.01_       | _93.15_        | 90.90             |
| M-Swin         | 83.58          | 91.05         | 92.08          | 90.05             |
| BAN            | 84.93          | 91.85         | 92.93          | 90.89             |
| **UA-BCD**     | **85.99**      | **92.47**     | **93.38**      | **91.57**         |
