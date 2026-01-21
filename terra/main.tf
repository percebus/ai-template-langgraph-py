
resource "random_id" "stack" {
  byte_length = 4
}

locals {
  environment       = var.ENVIRONMENT
  location          = var.ARM_REGION
  name_suffix       = var.ARM_RESOURCES_SUFFIX != "" ? var.ARM_RESOURCES_SUFFIX : random_id.stack.hex
  name_prefix_long  = "${var.ARM_RESOURCES_PREFIX}${var.REPOSITORY_NAME}-${local.environment}-${local.location}-${local.name_suffix}"
  name_prefix_short = "${local.environment}-${local.name_suffix}"
  repo_name         = var.REPOSITORY_NAME
  tags = {
    created_by  = local.repo_name
    environment = local.environment
  }
}

resource "azurerm_resource_group" "rg" {
  name     = "${local.name_prefix_long}-rg"
  location = local.location
}
