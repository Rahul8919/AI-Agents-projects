from langchain_core.tools import tool
@tool
def search_hotels_tool(city_code: str, check_in_date: str, check_out_date: str, adults: int = 1):
    """
    Searches for available hotel options in a specific city for given dates using Amadeus.
    Requires the IATA city code (e.g., 'PAR', 'BER') and dates in 'YYYY-MM-DD' format.
    Use get_current_date_tool first if dates are relative.
    """

    print(
        f"DEBUG: Calling Amadeus Hotel Search - City: {city_code}, "
        f"Check-in: {check_in_date}, Check-out: {check_out_date}, Adults: {adults}"
    )

    # Call Amadeus API - Hotel Search (find hotels by city)
    hotel_list_response = amadeus_client.reference_data.locations.hotels.by_city.get(
        cityCode=city_code,
        radius=50,
        radiusUnit="KM"
    )

    if not hotel_list_response.data or len(hotel_list_response.data) == 0:
        return f"No hotels found listed in Amadeus for city code {city_code}."

    # Get hotel IDs from the response (limit to first 5 for offers search)
    hotel_ids = [hotel["hotelId"] for hotel in hotel_list_response.data[:5]]

    # Now search for offers for these specific hotels
    hotel_offer_response = amadeus_client.shopping.hotel_offers_search.get(
        hotelIds=",".join(hotel_ids),
        checkInDate=check_in_date,
        checkOutDate=check_out_date,
        adults=adults
    )
    # Process the response (simplified)
    if hotel_offer_response.data and len(hotel_offer_response.data) > 0:
        results = []
        for offer in hotel_offer_response.data[:5]:  # Limit to showing 3 offers
            hotel_name = offer.get("hotel", {}).get("name", "N/A")
            price = offer.get("offers", [{}])[0].get("price", {}).get("total", "N/A")
            currency = offer.get("offers", [{}])[0].get("price", {}).get("currency", "")
            results.append(f"Hotel: {hotel_name}, Price: {price} {currency} (approx)")

        return "Found hotel options:\n- " + "\n- ".join(results)

    else:
        return f"No available hotel offers found for the dates in {city_code} among the checked hotels."