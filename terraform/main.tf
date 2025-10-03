provider "aws" {
  region = var.aws_region
}

resource "aws_instance" "rag_server" {
  ami           = var.ami_id
  instance_type = "t3.micro"
  
  root_block_device {
    volume_size = 30
  }

  user_data = templatefile("${path.module}/user-data.sh", {
    ecr_registry   = var.ecr_registry
    ecr_repository = var.ecr_repository
    image_tag      = var.image_tag
    region         = var.aws_region
  })

  tags = {
    Name = "rag-server"
  }
}

resource "aws_ebs_volume" "model_storage" {
  availability_zone = aws_instance.rag_server.availability_zone
  size              = 50
  type              = "gp3"
  
  tags = {
    Name = "ollama-models"
  }
}

resource "aws_volume_attachment" "model_attach" {
  device_name = "/dev/xvdf"
  volume_id   = aws_ebs_volume.model_storage.id
  instance_id = aws_instance.rag_server.id
}
