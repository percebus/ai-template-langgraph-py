terraform {
  required_providers {
    random = {
      source  = "hashicorp/random"
      version = "~>3.0"
    }

    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.106.1"
    }
  }

  required_version = "~> 1.14.3"
}

provider "azurerm" {
  features {}
}
