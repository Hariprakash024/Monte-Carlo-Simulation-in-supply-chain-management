terraform {
  required_version = ">= 1.3.0"
  required_providers {
    azurerm = { source = "hashicorp/azurerm", version = "~> 3.113" }
    random  = { source = "hashicorp/random",  version = "~> 3.6" }
    local   = { source = "hashicorp/local",   version = "~> 2.5" }
  }
}
provider "azurerm" { features {} }

