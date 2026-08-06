# Terraform infrastructure for the MLOps platform
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
    kubernetes = { source = "hashicorp/kubernetes", version = "~> 2.20" }
  }
}
provider "aws" { region = var.aws_region }
variable "aws_region" { default = "us-east-1" }
variable "project_name" { default = "mlops-platform" }
variable "environment" { default = "production" }
variable "mlflow_db_password" { sensitive = true }
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "${var.project_name}-${var.environment}-ml-artifacts"
  tags = { Name = "ML Artifacts", Environment = var.environment }
}
resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-${var.environment}-data"
  tags = { Name = "ML Data", Environment = var.environment }
}
resource "aws_ecr_repository" "api" {
  name = "${var.project_name}/api"
  image_tag_mutability = "MUTABLE"
  tags = { Environment = var.environment }
}
resource "aws_db_instance" "mlflow" {
  identifier = "${var.project_name}-mlflow"
  engine = "postgres"
  engine_version = "15"
  instance_class = "db.t3.micro"
  allocated_storage = 20
  db_name = "mlflow"
  username = "mlflow"
  password = var.mlflow_db_password
  skip_final_snapshot = true
  tags = { Environment = var.environment }
}
resource "aws_elasticache_cluster" "redis" {
  cluster_id = "${var.project_name}-redis"
  engine = "redis"
  node_type = "cache.t3.micro"
  num_cache_nodes = 1
  parameter_group_name = "default.redis7"
  tags = { Environment = var.environment }
}
output "s3_ml_artifacts" { value = aws_s3_bucket.ml_artifacts.bucket }
output "s3_data" { value = aws_s3_bucket.data.bucket }
output "ecr_repository" { value = aws_ecr_repository.api.repository_url }
output "mlflow_db_endpoint" { value = aws_db_instance.mlflow.endpoint }
output "redis_endpoint" { value = aws_elasticache_cluster.redis.cache_nodes[0].address }
