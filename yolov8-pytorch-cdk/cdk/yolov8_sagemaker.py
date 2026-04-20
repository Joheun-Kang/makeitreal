from constructs import Construct
from aws_cdk import (
    Aws,
    Stack,
    aws_s3 as s3,

    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_ec2 as ec2,
)
import aws_cdk as cdk

region = Aws.REGION
account = Aws.ACCOUNT_ID

# =============================================================================
# NOTEBOOK APPROACH (for development/experimentation)
# CDK Stack for
# 1. Create S3
# 2. Create SageMaker Notebook and use GitHub as Source
# =============================================================================

# class YOLOv8SageMakerStack(Stack):
#     """
#     The SageMaker Notebook is used to deploy the custom model on a SageMaker endpoint and test it.
#     """
#
#     def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
#         super().__init__(scope, construct_id, **kwargs)
#
#         ## Create S3 bucket
#         self.bucket = s3.Bucket(
#             self, "yolov8-s3",
#             auto_delete_objects=True,
#             removal_policy=cdk.RemovalPolicy.DESTROY)
#
#         ## IAM Roles
#         # Create role for Notebook instance
#         nRole = iam.Role(
#             self,
#             "yolov8-notebookAccessRole",
#             assumed_by=iam.ServicePrincipal('sagemaker'))
#
#         # Attach the right policies for SageMaker Notebook instance
#         nPolicy = iam.Policy(
#             self,
#             "yolov8-notebookAccessPolicy",
#             policy_name="yolov8-notebookAccessPolicy",
#             statements=[
#                 iam.PolicyStatement(actions=['sagemaker:*'], resources=['*']),
#                 iam.PolicyStatement(actions=['s3:ListAllMyBuckets'], resources=['arn:aws:s3:::*']),
#                 iam.PolicyStatement(actions=['iam:PassRole', 'ecr:*', "logs:*"], resources=['*', '*', '*']),
#                 iam.PolicyStatement(actions=['s3:*'], resources=[self.bucket.bucket_arn, self.bucket.bucket_arn+'/*']),
#                 ]).attach_to_role(nRole)
#
#         ## Create SageMaker Notebook instances cluster
#         nid = 'yolov8-sm-notebook'
#         notebook = sagemaker.CfnNotebookInstance(
#             self,
#             nid,
#             instance_type='ml.m5.4xlarge',
#             volume_size_in_gb=5,
#             notebook_instance_name=nid,
#             role_arn=nRole.role_arn,
#             additional_code_repositories=["https://github.com/aws-samples/host-yolov8-on-sagemaker-endpoint"],
#         )


# =============================================================================
# ENTERPRISE APPROACH (for production deployment)
# CDK Stack for
# 1. Create S3 bucket for model artifacts
# 2. Create IAM role for SageMaker
# 3. Deploy SageMaker Model, EndpointConfig, and Endpoint directly
# =============================================================================

YOLOV8_MODEL = "yolov8l.pt"
BUCKET_NAME = "yolov8-dev"
INSTANCE_TYPE = "ml.m5.4xlarge"

class YOLOv8SageMakerStack(Stack):
    """
    Deploys a YOLOv8 model directly to a SageMaker endpoint (no notebook required).
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ## 1. S3 bucket for model artifacts
        # self.bucket = s3.Bucket(
        #     self, "yolov8-s3",
        #     bucket_name=BUCKET_NAME,
        #     auto_delete_objects=True,
        #     removal_policy=cdk.RemovalPolicy.DESTROY)
        self.bucket = s3.Bucket.from_bucket_name(self, "yolov8-s3", BUCKET_NAME)

        ## 2. IAM Role for SageMaker endpoint
        role = iam.Role(
            self,
            "yolov8-endpointRole",
            assumed_by=iam.ServicePrincipal('sagemaker'))

        bucket_arn = f"arn:aws:s3:::{BUCKET_NAME}"
        policy = iam.Policy(
            self,
            "yolov8-endpointPolicy",
            policy_name="yolov8-endpointPolicy",
            statements=[
                iam.PolicyStatement(actions=['sagemaker:*'], resources=['*']),
                iam.PolicyStatement(actions=['s3:GetObject', 's3:ListBucket'],
                                    resources=[bucket_arn, bucket_arn+'/*']),
                iam.PolicyStatement(actions=['ecr:*', 'logs:*'], resources=['*']),
            ])
        policy.attach_to_role(role)

        ## 3. SageMaker Model
        model = sagemaker.CfnModel(
            self, "yolov8-model",
            execution_role_arn=role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:1.12-cpu-py38",
                model_data_url=f"s3://{BUCKET_NAME}/yolov8/model.tar.gz",
                environment={
                    "YOLOV8_MODEL": YOLOV8_MODEL,
                    "TS_MAX_RESPONSE_SIZE": "20000000",
                },
            ),
        )
        model.node.add_dependency(policy)

        ## 4. Endpoint Configuration
        endpoint_config = sagemaker.CfnEndpointConfig(
            self, "yolov8-endpoint-config",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    initial_instance_count=1,
                    instance_type=INSTANCE_TYPE,
                    model_name=model.attr_model_name,
                    variant_name="primary",
                )
            ],
        )

        ## 5. Endpoint
        endpoint = sagemaker.CfnEndpoint(
            self, "yolov8-endpoint",
            endpoint_config_name=endpoint_config.attr_endpoint_config_name,
        )

        ## 6. Output the endpoint name
        cdk.CfnOutput(self, "EndpointName",
                       value=endpoint.attr_endpoint_name,
                       description="SageMaker Endpoint Name")