variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1" # đổi nếu cần
}

variable "ami_id" {
  description = "AMI ID for the EC2 instance"
  type        = string
}

variable "ecr_registry" {
  description = "ECR registry (e.g., 601973176417.dkr.ecr.us-east-1.amazonaws.com)"
  type        = string
}

variable "ecr_repository" {
  description = "ECR repository name for the RAG backend image"
  type        = string
  default     = "rag_3"
}

variable "image_tag" {
  description = "Docker image tag for the RAG backend"
  type        = string
  default     = "latest"
}
