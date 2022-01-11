##### AWS SageMaker

delete role

delete notebook

delete training job

delete endpoint



##### Create Notebook Instance

Create new IAM Role.

Role ends with 843.

##### Refer create notebook instance 1.

##### attach polices to role.

Go to IAM, then Roles, then select your role, click attach policies - s3, apigateway

##### Refer attach policies to role

import sagemaker

from sagemaker.local import LocalSession

GitHub:

https://github.com/aws/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/pytorch_mnist



##### Notebook

modeltrainingpynb

```
2022-01-10 09:06:38 Uploading - Uploading generated training model
2022-01-10 09:06:38 Completed - Training job completed
ProfilerReport-1641805192: NoIssuesFound
Training seconds: 362
Billable seconds: 362
```

##### Both Training and Billable seconds are same!!!







