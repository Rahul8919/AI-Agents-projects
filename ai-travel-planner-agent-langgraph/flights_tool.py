# Service clients
from amadeus import Client, ResponseError

# Configure Amadeus Client
# We'll only initialize it if keys are provided, inside the tool later
amadeus_client = Client(
    client_id = amadeus_api_key,
    client_secret = amadeus_api_secret,
    hostname = "test",  # Start with the test environment
)

@tool
def search_flights_tool(
    origin_code: str,
    destination_code: str,
    departure_date: str,
    return_date: str | None = None,
    adults: int = 1,
    travel_class: str = "ECONOMY",
    currency: str = "USD",
    max_offers: int = 5,
):
    """
    Searches live flight prices and availability via Amadeus Flight Offers Search API.

    Required:
        origin_code, destination_code - IATA airport/city codes (e.g., 'YYZ', 'LHR')
        departure_date - 'YYYY-MM-DD'

    Optional:
        return_date - for round-trips; omit for one-way
        adults - number of adult passengers (default 1)
        travel_class - 'ECONOMY', 'PREMIUM_ECONOMY', 'BUSINESS', 'FIRST'
        currency - 3-letter code for pricing (default USD)
        max_offers - how many offers to list back
    """
    print(
    f"DEBUG: Calling Amadeus Flight Search - "
    f"{origin_code}->{destination_code}, "
    f"Depart {departure_date}, Return {return_date}, "
    f"Adults {adults}, Class {travel_class}"
)

# --- Call Amadeus Flight Offers Search API ---
    flight_search_params = {
        "originLocationCode": origin_code,
        "destinationLocationCode": destination_code,
        "departureDate": departure_date,
        "adults": adults,
        "travelClass": travel_class,
        "currencyCode": currency,
        "max": max_offers
    }

    if return_date:
        flight_search_params["returnDate"] = return_date

    response = amadeus_client.shopping.flight_offers_search.get(**flight_search_params)

# --- Parse the response ---
    if not response.data:
        return (
            f"No flight offers found for {origin_code} -> {destination_code} on "
            f"{departure_date}{' (return ' + return_date + ')' if return_date else ''}."
        )

    results = []
    for offer in response.data[:max_offers]:
        price = offer["price"]["total"]
        airline = offer["validatingAirlineCodes"][0]
        itinerary = offer["itineraries"][0]
        segments = itinerary["segments"]
        first_leg = segments[0]
        last_leg = segments[-1]

        dep_time = first_leg["departure"]["at"][:16].replace("T", " ")
        arr_time = last_leg["arrival"]["at"][:16].replace("T", " ")
        duration = itinerary["duration"].replace("PT", "")

        results.append(
            f"{airline} | {dep_time} -> {arr_time} | {duration} | {price} {currency}"
        )

        return "Found flight options:\n- " + "\n- ".join(results)