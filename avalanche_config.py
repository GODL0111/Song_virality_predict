"""
Avalanche network configuration and wallet setup for Song Virality Predictor
Reads configuration from environment variables for security
"""

import os
from dotenv import load_dotenv
from web3 import Web3

# Load environment variables
load_dotenv()

class AvalancheConfig:
    """Configuration class for Avalanche network and wallet settings"""
    
    # Network configurations
    NETWORKS = {
        'fuji': {
            'name': 'Avalanche Fuji Testnet',
            'rpc_url': 'https://api.avax-test.network/ext/bc/C/rpc',
            'chain_id': 43113,
            'currency_symbol': 'AVAX',
            'block_explorer': 'https://testnet.snowtrace.io'
        },
        'mainnet': {
            'name': 'Avalanche Mainnet',
            'rpc_url': 'https://api.avax.network/ext/bc/C/rpc',
            'chain_id': 43114,
            'currency_symbol': 'AVAX',
            'block_explorer': 'https://snowtrace.io'
        }
    }
    
    def __init__(self, network='fuji'):
        """
        Initialize Avalanche configuration
        
        Args:
            network (str): Network to use ('fuji' for testnet, 'mainnet' for mainnet)
        """
        self.network = network
        self.network_config = self.NETWORKS.get(network)
        
        if not self.network_config:
            raise ValueError(f"Unsupported network: {network}")
        
        # Load from environment variables
        self.private_key = os.getenv('AVALANCHE_PRIVATE_KEY')
        self.wallet_address = os.getenv('WALLET_ADDRESS')
        self.oracle_address = os.getenv('PREDICTION_ORACLE_ADDRESS')
        
        # Validate required environment variables
        self._validate_config()
        
        # Initialize Web3 connection
        self.web3 = None
        self._init_web3()
    
    def _validate_config(self):
        """Validate that required environment variables are set"""
        required_vars = ['AVALANCHE_PRIVATE_KEY', 'WALLET_ADDRESS']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                f"Please check your .env file."
            )
        
        # Validate private key format (should start with 0x and be 66 characters)
        if not self.private_key.startswith('0x') or len(self.private_key) != 66:
            raise ValueError(
                "AVALANCHE_PRIVATE_KEY must be a valid private key starting with 0x and 66 characters long"
            )
        
        # Validate wallet address format
        if not Web3.is_address(self.wallet_address):
            raise ValueError("WALLET_ADDRESS must be a valid Ethereum address")
    
    def _init_web3(self):
        """Initialize Web3 connection to Avalanche network"""
        try:
            self.web3 = Web3(Web3.HTTPProvider(self.network_config['rpc_url']))
            
            # Test connection
            if not self.web3.is_connected():
                raise ConnectionError(f"Failed to connect to {self.network_config['name']}")
                
            print(f"✅ Connected to {self.network_config['name']}")
            
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Web3 connection: {str(e)}")
    
    def get_web3(self):
        """Get Web3 instance"""
        return self.web3
    
    def get_account(self):
        """Get account from private key"""
        if not self.web3:
            raise RuntimeError("Web3 not initialized")
        
        account = self.web3.eth.account.from_key(self.private_key)
        return account
    
    def get_network_info(self):
        """Get current network information"""
        return {
            'network': self.network,
            'name': self.network_config['name'],
            'chain_id': self.network_config['chain_id'],
            'rpc_url': self.network_config['rpc_url'],
            'currency': self.network_config['currency_symbol'],
            'explorer': self.network_config['block_explorer']
        }
    
    def get_balance(self, address=None):
        """
        Get AVAX balance for an address
        
        Args:
            address (str): Address to check balance for (defaults to wallet address)
            
        Returns:
            float: Balance in AVAX
        """
        if not address:
            address = self.wallet_address
            
        balance_wei = self.web3.eth.get_balance(address)
        balance_avax = self.web3.from_wei(balance_wei, 'ether')
        return float(balance_avax)
    
    def estimate_gas_price(self):
        """
        Get current gas price estimate
        
        Returns:
            int: Gas price in wei
        """
        return self.web3.eth.gas_price
    
    def get_transaction_receipt(self, tx_hash):
        """
        Get transaction receipt by hash
        
        Args:
            tx_hash (str): Transaction hash
            
        Returns:
            dict: Transaction receipt
        """
        return self.web3.eth.get_transaction_receipt(tx_hash)
    
    def get_block_explorer_url(self, tx_hash=None, address=None):
        """
        Get block explorer URL for transaction or address
        
        Args:
            tx_hash (str): Transaction hash
            address (str): Address
            
        Returns:
            str: Block explorer URL
        """
        base_url = self.network_config['block_explorer']
        
        if tx_hash:
            return f"{base_url}/tx/{tx_hash}"
        elif address:
            return f"{base_url}/address/{address}"
        else:
            return base_url
    
    def wait_for_transaction(self, tx_hash, timeout=120):
        """
        Wait for transaction to be mined
        
        Args:
            tx_hash (str): Transaction hash
            timeout (int): Timeout in seconds
            
        Returns:
            dict: Transaction receipt
        """
        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            return receipt
        except Exception as e:
            raise RuntimeError(f"Transaction failed or timed out: {str(e)}")


def get_avalanche_config(network='fuji'):
    """
    Factory function to get Avalanche configuration
    
    Args:
        network (str): Network to use ('fuji' or 'mainnet')
        
    Returns:
        AvalancheConfig: Configured instance
    """
    return AvalancheConfig(network)


def validate_environment():
    """
    Validate that all required environment variables are set
    
    Returns:
        bool: True if valid, raises exception if not
    """
    try:
        config = AvalancheConfig()
        print("✅ Environment validation successful")
        return True
    except Exception as e:
        print(f"❌ Environment validation failed: {str(e)}")
        raise


if __name__ == "__main__":
    """Test the configuration"""
    try:
        # Test Fuji testnet configuration
        config = get_avalanche_config('fuji')
        
        print("\n🔧 Avalanche Configuration Test")
        print("=" * 40)
        
        # Display network info
        network_info = config.get_network_info()
        print(f"Network: {network_info['name']}")
        print(f"Chain ID: {network_info['chain_id']}")
        print(f"RPC URL: {network_info['rpc_url']}")
        
        # Check balance
        balance = config.get_balance()
        print(f"Wallet Balance: {balance:.4f} AVAX")
        
        # Check gas price
        gas_price = config.estimate_gas_price()
        gas_price_gwei = config.web3.from_wei(gas_price, 'gwei')
        print(f"Current Gas Price: {gas_price_gwei:.2f} Gwei")
        
        print("\n✅ Configuration test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Configuration test failed: {str(e)}")
        print("\nPlease check your .env file and ensure all required variables are set.")