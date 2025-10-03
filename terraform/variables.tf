variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "ap-southeast-1" # Singapore
}

variable "ami_id" {
  description = "AMI ID for the EC2 instance"
  type        = string
}

variable "ecr_repository" {
  description = "ECR repository name for the RAG backend image"
  type        = string
}

variable "image_tag" {
  description = "Docker image tag for the RAG backend"
  type        = string
  default     = "latest"
}
