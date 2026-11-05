# First SageMaker training job

Keep all three Python files in this folder. `train.py` is the previously delivered
training implementation, copied without modelling changes. Run the launcher from
your existing authenticated SageMaker terminal or an AWS-configured local terminal.

1. Install the launcher dependency: `python -m pip install boto3`.
2. Choose an existing same-region S3 bucket, SageMaker execution-role ARN, and
   compatible training image URI. The image needs Python 3.9+ and installed numpy,
   pandas, scikit-learn >=1.2, and joblib. This launcher does not install packages
   inside the job. For reproducibility use an immutable image digest or fixed tag
   with dependency versions matching your validated environment.
3. Run the command below with actual values and the input used for local training.
   The example cutoff creates the validation candidate; it does not select a new cutoff.

```bash
python launch_training.py --data ../data/processed/product_22041_daily_sales.csv --region YOUR_REGION --bucket YOUR_BUCKET --role-arn YOUR_EXECUTION_ROLE_ARN --image-uri YOUR_TRAINING_IMAGE_URI --train-end 2011-05-09 --product-id 22041 --alpha 0.01 --wait
```

Use the correct path to your CSV. For the earlier feature CSV instead, add
`--sales-col target_sales`. Preserve your locally validated training input.
Append `--dry-run` to inspect the request without uploads or a job submission.
A real submission provisions billable training compute, with one instance and a
900-second training-runtime limit. Provisioning/upload overhead is additional.
No endpoint is created. Instance availability and quotas depend on your account.

The launcher uploads data and source files into a fresh job-specific S3 prefix.
SageMaker File-mode channels provide them under `/opt/ml/input/data/train` and
`/opt/ml/input/data/code`. A custom entrypoint calls the validated training function;
SageMaker collects `/opt/ml/model` into its model artifact archive on S3.
Network isolation is enabled; dependencies must already be present in the image.

The caller needs S3 upload permissions, SageMaker create/describe training-job
permissions, and iam:PassRole for the execution role. That role must trust
sagemaker.amazonaws.com and have applicable S3 read/write, ECR image-pull and
CloudWatch logging permissions. Encrypted resources may need KMS access as well.
Use your already configured role if it has these permissions.

After completion, the launcher prints the model artifact S3 URI. Download its
model.tar.gz using the AWS console or `aws s3 cp`, and extract into a fresh local
folder. It contains direct_ridge.joblib, metadata.json, training_audit.csv and
training_history.csv. Check metadata.json for the container's dependency versions;
use the same scikit-learn version for evaluate.py. Run your existing evaluate.py
against this cloud-trained package and the same validation data/date interval.
This training job does not execute preparation, feature-table generation,
evaluation, or inventory simulation. Those are subsequent pipeline steps.

Validation performed here: launcher request structure and mocked submission,
entrypoint integration with a local simulated SageMaker directory tree, and Python
syntax. A real AWS job and your selected image still require account-side validation.
