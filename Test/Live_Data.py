import requests

def get_live_crypto_data(coin_ids):
    """
    Fetch live market data for multiple cryptocurrencies using CoinGecko API.

    Parameters:
        coin_ids (list): List of cryptocurrency IDs as per CoinGecko (e.g., ['bitcoin', 'ethereum'])

    Returns:
        dict: Dictionary with coin names as keys and market data as values
    """

    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ','.join(coin_ids),
        "price_change_percentage": "1h,24h,7d"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        all_data = {}

        for coin in data:
            coin_name = coin["id"]
            info = {
                "price": coin["current_price"],
                "1h": coin.get("price_change_percentage_1h_in_currency", 0),
                "24h": coin.get("price_change_percentage_24h_in_currency", 0),
                "7d": coin.get("price_change_percentage_7d_in_currency", 0),
                "24h_volume": coin["total_volume"],
                "market_cap": coin["market_cap"],
                "volatility (abs 24h)": abs(coin.get("price_change_percentage_24h_in_currency", 0))
            }

            all_data[coin_name] = info

        # Print nicely
        for name, metrics in all_data.items():
            print(f"\n {name.upper()} Market Data:")
            for k, v in metrics.items():
                print(f"{k}: {v}")

        return all_data

    except Exception as e:
        print(f" Error fetching data: {e}")
        return {}

# Example usage
crypto_list = ["bitcoin", "ethereum", "solana", "dogecoin"]
get_live_crypto_data(crypto_list)
