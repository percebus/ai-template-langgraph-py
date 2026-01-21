# IaC

## terra/ folder

Having everything under `terra` allows to simplify devops activities, such as

```script
$> terraform -chdir=./terra fmt -recursive -diff -list=true -check
```

### Why "terra" and not "terraform"?

Well, there is a `terragrunt` tool, and also a `terratester` tool.

It felt like a good idea to have a single folder for all of the `terra`form-ish related stuff.

### Project folders

More often than not, a single terraform project would suffice for all of your needs.

But sometimes, we might want to separate into separate projects things like

- Azure policies
- Network: VNets, peerings, subnets, NSGs, etc.
- Foundry stuff: OpenAI service, deployments, etc.

When someone encounters this, is VERY frustrating to have to scaffold YET ANOTHER terraform project.

This mono-repo allows to have a single terraform 'solution' with multiple projects.

### terraform CLI

1. `cd` into `terra/{your-project}`
1. `terraform init`

## References

- [Terraform > Azure > Build](https://developer.hashicorp.com/terraform/tutorials/azure-get-started/azure-build)
