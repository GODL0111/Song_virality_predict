"""
Deployment script for MusicPredictionOracle contract on Avalanche
Provides instructions and utilities for deploying the contract using Foundry or Hardhat
"""

import os
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ContractDeployer:
    """Utility class for deploying the MusicPredictionOracle contract"""
    
    def __init__(self, network='fuji'):
        """
        Initialize the deployer
        
        Args:
            network (str): Target network ('fuji' for testnet, 'mainnet' for mainnet)
        """
        self.network = network
        self.networks = {
            'fuji': {
                'name': 'Avalanche Fuji Testnet',
                'rpc_url': 'https://api.avax-test.network/ext/bc/C/rpc',
                'chain_id': 43113,
                'explorer': 'https://testnet.snowtrace.io'
            },
            'mainnet': {
                'name': 'Avalanche Mainnet',
                'rpc_url': 'https://api.avax.network/ext/bc/C/rpc',
                'chain_id': 43114,
                'explorer': 'https://snowtrace.io'
            }
        }
        
        self.private_key = os.getenv('AVALANCHE_PRIVATE_KEY')
        if not self.private_key:
            raise ValueError("AVALANCHE_PRIVATE_KEY not found in environment variables")
    
    def get_foundry_deploy_command(self):
        """
        Generate Foundry deployment command
        
        Returns:
            str: Complete forge create command
        """
        network_config = self.networks[self.network]
        
        command = f"""forge create \\
    --rpc-url {network_config['rpc_url']} \\
    --private-key {self.private_key} \\
    contracts/MusicPredictionOracle.sol:MusicPredictionOracle \\
    --legacy"""
        
        return command
    
    def get_hardhat_config(self):
        """
        Generate Hardhat network configuration
        
        Returns:
            str: Hardhat network configuration for hardhat.config.js
        """
        network_config = self.networks[self.network]
        
        config = f"""
// Add this to your hardhat.config.js networks section
{self.network}: {{
  url: "{network_config['rpc_url']}",
  accounts: ["{self.private_key}"],
  chainId: {network_config['chain_id']},
  gasPrice: "auto",
  gas: "auto"
}}"""
        
        return config
    
    def get_hardhat_deploy_script(self):
        """
        Generate Hardhat deployment script
        
        Returns:
            str: Complete deployment script for Hardhat
        """
        script = f"""
// Save this as scripts/deploy-oracle.js
const {{ ethers }} = require("hardhat");

async function main() {{
  console.log("Deploying MusicPredictionOracle to {self.networks[self.network]['name']}...");
  
  // Get the contract factory
  const MusicPredictionOracle = await ethers.getContractFactory("MusicPredictionOracle");
  
  // Deploy the contract
  const oracle = await MusicPredictionOracle.deploy();
  
  // Wait for deployment
  await oracle.deployed();
  
  console.log("MusicPredictionOracle deployed to:", oracle.address);
  console.log("Transaction hash:", oracle.deployTransaction.hash);
  console.log("Explorer URL: {self.networks[self.network]['explorer']}/address/" + oracle.address);
  
  // Verify contract on explorer (if supported)
  if (network.name !== "hardhat" && network.name !== "localhost") {{
    console.log("Waiting for block confirmations...");
    await oracle.deployTransaction.wait(6);
    
    console.log("Verifying contract...");
    try {{
      await hre.run("verify:verify", {{
        address: oracle.address,
        constructorArguments: [],
      }});
    }} catch (error) {{
      console.log("Verification failed:", error.message);
    }}
  }}
}}

main()
  .then(() => process.exit(0))
  .catch((error) => {{
    console.error(error);
    process.exit(1);
  }});
"""
        
        return script
    
    def print_deployment_instructions(self):
        """Print complete deployment instructions"""
        print(f"🚀 MusicPredictionOracle Deployment Instructions")
        print(f"Target Network: {self.networks[self.network]['name']}")
        print("=" * 60)
        
        print("\n📋 Prerequisites:")
        print("1. Set AVALANCHE_PRIVATE_KEY in your .env file")
        print("2. Ensure your wallet has sufficient AVAX for gas fees")
        print("3. Install Foundry or Hardhat")
        
        print("\n⚒️  Option 1: Deploy with Foundry")
        print("-" * 30)
        print("1. Install Foundry: https://book.getfoundry.sh/getting-started/installation")
        print("2. Initialize a Foundry project (if not already done):")
        print("   forge init --no-git")
        print("3. Copy the contract to src/MusicPredictionOracle.sol")
        print("4. Run the deployment command:")
        print(f"\n{self.get_foundry_deploy_command()}")
        
        print("\n🔨 Option 2: Deploy with Hardhat")
        print("-" * 30)
        print("1. Install Hardhat: npm install --save-dev hardhat")
        print("2. Initialize Hardhat project: npx hardhat")
        print("3. Add network configuration to hardhat.config.js:")
        print(self.get_hardhat_config())
        print("\n4. Create deployment script:")
        print(self.get_hardhat_deploy_script())
        print("\n5. Run deployment:")
        print(f"   npx hardhat run scripts/deploy-oracle.js --network {self.network}")
        
        print("\n✅ After Deployment:")
        print("1. Copy the deployed contract address")
        print("2. Set PREDICTION_ORACLE_ADDRESS in your .env file")
        print("3. Verify the contract on the block explorer")
        print(f"4. Explorer: {self.networks[self.network]['explorer']}")
        
        print("\n🔍 Verification:")
        print("- Check the contract on the block explorer")
        print("- Test basic contract functions")
        print("- Run the AvalancheSongPredictor demo to ensure integration works")
    
    def create_foundry_project_structure(self):
        """Create basic Foundry project structure"""
        try:
            # Create directories
            os.makedirs("src", exist_ok=True)
            os.makedirs("script", exist_ok=True)
            os.makedirs("test", exist_ok=True)
            
            # Copy contract to src directory
            if os.path.exists("contracts/MusicPredictionOracle.sol"):
                import shutil
                shutil.copy2("contracts/MusicPredictionOracle.sol", "src/MusicPredictionOracle.sol")
                print("✅ Contract copied to src/MusicPredictionOracle.sol")
            
            # Create foundry.toml if it doesn't exist
            if not os.path.exists("foundry.toml"):
                foundry_config = """[profile.default]
src = "src"
out = "out"
libs = ["lib"]
solc_version = "0.8.19"

[rpc_endpoints]
fuji = "https://api.avax-test.network/ext/bc/C/rpc"
mainnet = "https://api.avax.network/ext/bc/C/rpc"
"""
                with open("foundry.toml", "w") as f:
                    f.write(foundry_config)
                print("✅ Created foundry.toml configuration")
            
            print("✅ Foundry project structure created")
            
        except Exception as e:
            print(f"❌ Failed to create Foundry project structure: {str(e)}")
    
    def verify_prerequisites(self):
        """Verify deployment prerequisites"""
        print("🔍 Verifying deployment prerequisites...")
        
        issues = []
        
        # Check private key
        if not self.private_key:
            issues.append("AVALANCHE_PRIVATE_KEY not set in environment")
        elif len(self.private_key) != 66 or not self.private_key.startswith('0x'):
            issues.append("AVALANCHE_PRIVATE_KEY format is invalid")
        
        # Check contract file
        if not os.path.exists("contracts/MusicPredictionOracle.sol"):
            issues.append("MusicPredictionOracle.sol not found in contracts directory")
        
        # Check for deployment tools
        foundry_available = False
        hardhat_available = False
        
        try:
            subprocess.run(["forge", "--version"], capture_output=True, check=True)
            foundry_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        try:
            subprocess.run(["npx", "hardhat", "--version"], capture_output=True, check=True)
            hardhat_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        if not foundry_available and not hardhat_available:
            issues.append("Neither Foundry nor Hardhat is installed")
        
        if issues:
            print("❌ Prerequisites check failed:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("✅ All prerequisites met")
            if foundry_available:
                print("   - Foundry available")
            if hardhat_available:
                print("   - Hardhat available")
            return True


def main():
    """Main deployment helper function"""
    print("🔧 MusicPredictionOracle Deployment Helper")
    print("=" * 50)
    
    # Get network choice
    network = input("Choose network (fuji/mainnet) [fuji]: ").strip().lower() or 'fuji'
    
    if network not in ['fuji', 'mainnet']:
        print("❌ Invalid network. Choose 'fuji' or 'mainnet'")
        return
    
    if network == 'mainnet':
        confirm = input("⚠️  You are deploying to MAINNET. Are you sure? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Deployment cancelled")
            return
    
    try:
        deployer = ContractDeployer(network)
        
        # Verify prerequisites
        if not deployer.verify_prerequisites():
            print("\n❌ Please fix the issues above before proceeding")
            return
        
        # Print deployment instructions
        deployer.print_deployment_instructions()
        
        # Offer to create Foundry project structure
        create_foundry = input("\nCreate Foundry project structure? (y/n) [n]: ").strip().lower()
        if create_foundry == 'y':
            deployer.create_foundry_project_structure()
        
        print(f"\n🎯 Next Steps:")
        print("1. Choose your preferred deployment method (Foundry or Hardhat)")
        print("2. Follow the instructions above")
        print("3. Update your .env file with the deployed contract address")
        print("4. Test the integration with AvalancheSongPredictor")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()