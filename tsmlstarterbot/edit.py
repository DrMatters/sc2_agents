def parse(all_games_json_data, bot_to_imitate=None, dump_features_location=None, games_num_players=None):
    """
    Parse the games to compute features. This method computes PER_PLANET_FEATURES features for each planet in each frame
    in each game the bot we're imitating played.

    :param all_games_json_data: list of json dictionaries describing games
    :param bot_to_imitate: name of the bot to imitate or None if we want to imitate the bot who won the most games
    :param dump_features_location: location where to serialize the features
    :return: data ready for training
    """
    print("Parsing data...")

    parsed_games = 0
    skipped_n_players_games = 0

    training_data = []

    if bot_to_imitate is None:
        print("No bot name provided, choosing the bot with the highest number of games won...")
        players_games_count = {}
        for json_data in all_games_json_data:
            w = find_winner(json_data)
            p = json_data['player_names'][int(w)]
            if p not in players_games_count:
                players_games_count[p] = 0
            players_games_count[p] += 1

        bot_to_imitate = max(players_games_count, key=players_games_count.get)
    print("Bot to imitate: {}.".format(bot_to_imitate))

    for json_data in all_games_json_data:

        if games_num_players:
            if json_data['num_players'] != games_num_players:
                skipped_n_players_games += 1
                continue

        frames = json_data['frames']
        moves = json_data['moves']
        width = json_data['width']
        height = json_data['height']

        # For each game see if bot_to_imitate played in it
        if bot_to_imitate not in set(json_data['player_names']):
            continue
        # We train on all the games of the bot regardless whether it won or not.
        bot_to_imitate_id = str(json_data['player_names'].index(bot_to_imitate))

        parsed_games = parsed_games + 1
        game_training_data = []

        # Ignore the last frame, no decision to be made there
        for idx in range(len(frames) - 1):

            current_moves = moves[idx]
            current_frame = frames[idx]

            if bot_to_imitate_id not in current_frame['ships'] or len(current_frame['ships'][bot_to_imitate_id]) == 0:
                continue

            planet_features = {}  # planet_id -> list of features per ship per planet
            current_planets = current_frame['planets']

            # find % allocation for all ships
            all_moving_ships = 0
            allocations = {}

            # for each planet we want to find how many ships are being moved towards it now
            for ship_id, ship_data in current_frame['ships'][bot_to_imitate_id].items():
                if ship_id in current_moves[bot_to_imitate_id][0]:
                    p = find_target_planet(bot_to_imitate_id, current_frame,
                                           json_data['planets'],
                                           current_moves[bot_to_imitate_id][0][ship_id],
                                           )
                    planet_id = int(p)
                    if planet_id < 0 or planet_id >= PLANET_MAX_NUM:
                        continue

                    if p not in allocations:
                        allocations[p] = 0
                    allocations[p] = allocations[p] + 1
                    all_moving_ships = all_moving_ships + 1

            if all_moving_ships == 0:
                continue

            # Compute what % of the ships should be sent to given planet
            for planet_id, allocated_ships in allocations.items():
                allocations[planet_id] = allocated_ships / all_moving_ships

            # Compute features
            for planet_id in range(PLANET_MAX_NUM):

                if str(planet_id) not in current_planets:
                    continue
                planet_data = current_planets[str(planet_id)]

                gravity = 0
                planet_base_data = json_data['planets'][planet_id]
                closest_friendly_ship_distance = 10000
                closest_enemy_ship_distance = 10000

                ownership = 0
                if str(planet_data['owner']) == bot_to_imitate_id:
                    ownership = 1
                elif planet_data['owner'] is not None:
                    ownership = -1

                average_distance = 0
                my_ships_health = 0

                for player_id, ships in current_frame['ships'].items():
                    for ship_id, ship_data in ships.items():
                        is_bot_to_imitate = 1 if player_id == bot_to_imitate_id else -1
                        dist2 = distance2(planet_base_data['x'], planet_base_data['y'], ship_data['x'], ship_data['y'])
                        dist = math.sqrt(dist2)
                        gravity = gravity + is_bot_to_imitate * ship_data['health'] / dist2
                        if is_bot_to_imitate == 1:
                            closest_friendly_ship_distance = min(closest_friendly_ship_distance, dist)
                            average_distance = average_distance + dist * ship_data['health']
                            my_ships_health = my_ships_health + ship_data['health']
                        else:
                            closest_enemy_ship_distance = min(closest_enemy_ship_distance, dist)

                distance_from_center = distance(planet_base_data['x'], planet_base_data['y'], width / 2, height / 2)
                average_distance = average_distance / my_ships_health

                is_active = 1.0 if planet_base_data['docking_spots'] > len(
                    planet_data['docked_ships']) or ownership != 1 else 0.0

                signed_current_production = planet_data['current_production'] * ownership

                # Features of the planet are inserted into the vector in the order described by FEATURE_NAMES
                planet_features[str(planet_id)] = [
                    planet_data['health'],
                    planet_base_data['docking_spots'] - len(planet_data['docked_ships']),
                    planet_data['remaining_production'],
                    signed_current_production,
                    gravity,
                    closest_friendly_ship_distance,
                    closest_enemy_ship_distance,
                    ownership,
                    distance_from_center,
                    average_distance,
                    is_active]

            game_training_data.append((planet_features, allocations))
        training_data.append(game_training_data)

    if parsed_games == 0:
        raise Exception("Didn't find any matching games. Try different bot.")

    if dump_features_location is not None:
        serialize_data(training_data, dump_features_location)

    flat_training_data = [item for sublist in training_data for item in sublist]

    print(f"Data parsed, parsed {parsed_games} games, skipped {skipped_n_players_games} by n_players,"
          f" total frames: {len(flat_training_data)}")

    return format_data_for_training(flat_training_data)
