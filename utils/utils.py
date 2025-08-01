#from pycragapi import CRAG

#api = CRAG()

import json

open_search_entity_by_name = {
            "type": "function",
            "function": {
                "name": "open_search_entity_by_name",
                "description": "Search for entities by name in the Open domain.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The name of the entity interested.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

open_get_entity = {
            "type": "function",
            "function": {
                "name": "open_get_entity",
                "description": "Retrieve detailed information about an entity in the Open domain.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "The name of the entity interested.",
                        }
                    },
                    "required": ["entity"],
                },
            },
        }

movie_get_person_info = {
            "type": "function",
            "function": {
                "name": "movie_get_person_info",
                "description": "Get information about a person related to movies.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "person_name": {
                            "type": "string",
                            "description": "The name of the person interested.",
                        }
                    },
                    "required": ["person_name"],
                },
            },
        }

movie_get_movie_info = {
            "type": "function",
            "function": {
                "name": "movie_get_movie_info",
                "description": "Get information about a movie.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "movie_name": {
                            "type": "string",
                            "description": "The name of the movie interested.",
                        }
                    },
                    "required": ["movie_name"],
                },
            },
        }

movie_get_year_info = {
            "type": "function",
            "function": {
                "name": "movie_get_year_info",
                "description": "Get information about movies released in a specific year.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "year": {
                            "type": "string",
                            "description": "The year interested.",
                        }
                    },
                    "required": ["year"],
                },
            },
        }

movie_get_movie_info_by_id = {
            "type": "function",
            "function": {
                "name": "movie_get_movie_info_by_id",
                "description": "Get movie information by its unique ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "movie_id": {
                            "type": "integer",
                            "description": "The movie id interested.",
                        }
                    },
                    "required": ["movie_id"],
                },
            },
        }

movie_get_person_info_by_id = {
            "type": "function",
            "function": {
                "name": "movie_get_person_info_by_id",
                "description": "Get person information by their unique ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "person_id": {
                            "type": "integer",
                            "description": "The person id interested.",
                        }
                    },
                    "required": ["person_id"],
                },
            },
        }

finance_get_company_name = {
            "type": "function",
            "function": {
                "name": "finance_get_company_name",
                "description": "Given ticker name, search for company names in the finance domain.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query interested.",
                        }
                    },
                    "required": ["query"],
                },
            },
}

finance_get_ticker_by_name = {
            "type": "function",
            "function": {
                "name": "finance_get_ticker_by_name",
                "description": "Retrieve the ticker symbol for a given company name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query interested.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

finance_get_price_history = {
            "type": "function",
            "function": {
                "name": "finance_get_price_history",
                "description": "Given ticker name, return daily Open price, Close price, High price, Low price and trading Volume.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_name": {
                            "type": "string",
                            "description": "The ticker name of the stock interested.",
                        }
                    },
                    "required": ["ticker_name"],
                },
            },
        }

finance_get_detailed_price_history = {
            "type": "function",
            "function": {
                "name": "finance_get_detailed_price_history",
                "description": "Given ticker name, return minute-level Open price, Close price, High price, Low price and trading Volume.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_name": {
                            "type": "string",
                            "description": "The ticker name of the stock interested.",
                        }
                    },
                    "required": ["ticker_name"],
                },
            },
        }

finance_get_dividends_history = {
            "type": "function",
            "function": {
                "name": "finance_get_dividends_history",
                "description": "Given ticker name, return dividend history.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_name": {
                            "type": "string",
                            "description": "The ticker name of the stock interested.",
                        }
                    },
                    "required": ["ticker_name"],
                },
            },
        }

finance_get_market_capitalization = {
            "type": "function",
            "function": {
                "name": "finance_get_market_capitalization",
                "description": "Given ticker name, return the market capitalization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_name": {
                            "type": "string",
                            "description": "The ticker name of the stock interested.",
                        }
                    },
                    "required": ["ticker_name"],
                },
            },
        }

finance_get_eps = {
            "type": "function",
            "function": {
                "name": "finance_get_eps",
                "description": "Given ticker name, return earnings per share.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_name": {
                            "type": "string",
                            "description": "The ticker name of the stock interested.",
                        }
                    },
                    "required": ["ticker_name"],
                },
            },
        }

finance_get_pe_ratio = {
            "type": "function",
            "function": {
                "name": "finance_get_pe_ratio",
                "description": "Given ticker name, return price-to-earnings ratio.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_name": {
                            "type": "string",
                            "description": "The ticker name of the stock interested.",
                        }
                    },
                    "required": ["ticker_name"],
                },
            },
        }

finance_get_info = {
            "type": "function",
            "function": {
                "name": "finance_get_info",
                "description": "Given ticker name, return rough meta data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_name": {
                            "type": "string",
                            "description": "The ticker name of the stock interested.",
                        }
                    },
                    "required": ["ticker_name"],
                },
            },
        }

music_get_members = {
            "type": "function",
            "function": {
                "name": "music_get_members",
                "description": "Return the member list of a band / person.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "band_name": {
                            "type": "string",
                            "description": "The name of the band / person interested.",
                        }
                    },
                    "required": ["band_name"],
                },
            },
        }

music_get_artist_birth_date = {
            "type": "function",
            "function": {
                "name": "music_get_artist_birth_date",
                "description": "Return the birth date of the artist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artist_name": {
                            "type": "string",
                            "description": "The name of the artist interested.",
                        }
                    },
                    "required": ["artist_name"],
                },
            },
        }

music_get_artist_birth_place = {
            "type": "function",
            "function": {
                "name": "music_get_artist_birth_place",
                "description": "Return the birth place country code (2-digit) for the input artist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artist_name": {
                            "type": "string",
                            "description": "The name of the artist interested.",
                        }
                    },
                    "required": ["artist_name"],
                },
            },
        }

music_get_lifespan = {
            "type": "function",
            "function": {
                "name": "music_get_lifespan",
                "description": "Return the lifespan of the artist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artist_name": {
                            "type": "string",
                            "description": "The name of the artist interested.",
                        }
                    },
                    "required": ["artist_name"],
                },
            },
        }

music_get_artist_all_works = {
            "type": "function",
            "function": {
                "name": "music_get_artist_all_works",
                "description": "Return the list of all works of a certain artist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artist_name": {
                            "type": "string",
                            "description": "The name of the artist interested.",
                        }
                    },
                    "required": ["artist_name"],
                },
            },
        }

music_get_song_author = {
            "type": "function",
            "function": {
                "name": "music_get_song_author",
                "description": "Get the author of a song.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "song_name": {
                            "type": "string",
                            "description": "The name of the song interested.",
                        }
                    },
                    "required": ["song_name"],
                },
            },
        }

music_get_song_release_country = {
            "type": "function",
            "function": {
                "name": "music_get_song_release_country",
                "description": "Get the release country of a song.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "song_name": {
                            "type": "string",
                            "description": "The name of the song interested.",
                        }
                    },
                    "required": ["song_name"],
                },
            },
        }

music_get_song_release_date = {
            "type": "function",
            "function": {
                "name": "music_get_song_release_date",
                "description": "Get the release date of a song.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "song_name": {
                            "type": "string",
                            "description": "The name of the song interested.",
                        }
                    },
                    "required": ["song_name"],
                },
            },
        }

music_search_artist_entity_by_name = {
            "type": "function",
            "function": {
                "name": "music_search_artist_entity_by_name",
                "description": "Search for music artists by name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artist_name": {
                            "type": "string",
                            "description": "The name of the artist interested.",
                        }
                    },
                    "required": ["artist_name"],
                },
            },
        }

music_search_song_entity_by_name = {
            "type": "function",
            "function": {
                "name": "music_search_song_entity_by_name",
                "description": "Search for songs by name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "song_name": {
                            "type": "string",
                            "description": "The name of the song interested.",
                        }
                    },
                    "required": ["song_name"],
                },
            },
        }

music_get_billboard_rank_date = {
            "type": "function",
            "function": {
                "name": "music_get_billboard_rank_date",
                "description": "Get Billboard ranking for a specific rank and date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rank": {
                            "type": "integer",
                            "description": "The rank you are interested.",
                        },
                        "date": {
                            "type": "string",
                            "description": "The date you are interested.",
                        }
                    },
                    "required": ["rank"],
                },
            },
        }

music_get_billboard_attributes = {
            "type": "function",
            "function": {
                "name": "music_get_billboard_attributes",
                "description": "Get attributes of a song from Billboard rankings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "The date you are interested.",
                        },
                        "attribute": {
                            "type": "string",
                            "description": "The attribute you are interested.",
                        },
                        "song_name": {
                            "type": "string",
                            "description": "The song you are interested.",
                        }
                    },
                    "required": ["date", "attribute", "song_name"],
                },
            },
        }

music_grammy_get_best_artist_by_year = {
            "type": "function",
            "function": {
                "name": "music_grammy_get_best_artist_by_year",
                "description": "Get the Grammy Best New Artist for a specific year.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "year": {
                            "type": "integer",
                            "description": "The year interested.",
                        }
                    },
                    "required": ["year"],
                },
            },
        }

music_grammy_get_award_count_by_artist = {
            "type": "function",
            "function": {
                "name": "music_grammy_get_award_count_by_artist",
                "description": "Get the total Grammy awards won by an artist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artist_name": {
                            "type": "string",
                            "description": "The artist interested.",
                        }
                    },
                    "required": ["artist_name"],
                },
            },
        }

music_grammy_get_award_count_by_song = {
            "type": "function",
            "function": {
                "name": "music_grammy_get_award_count_by_song",
                "description": "Get the total Grammy awards won by a song.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "song_name": {
                            "type": "string",
                            "description": "The song interested.",
                        }
                    },
                    "required": ["song_name"],
                },
            },
        }

music_grammy_get_best_song_by_year = {
            "type": "function",
            "function": {
                "name": "music_grammy_get_best_song_by_year",
                "description": "Get the Grammy Song of the Year for a specific year.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "year": {
                            "type": "integer",
                            "description": "The year interested.",
                        }
                    },
                    "required": ["year"],
                },
            },
        }

music_grammy_get_award_date_by_artist = {
            "type": "function",
            "function": {
                "name": "music_grammy_get_award_date_by_artist",
                "description": "Get the years an artist won a Grammy award.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artist_name": {
                            "type": "string",
                            "description": "The artist interested.",
                        }
                    },
                    "required": ["artist_name"],
                },
            },
        }

music_grammy_get_best_album_by_year = {
            "type": "function",
            "function": {
                "name": "music_grammy_get_best_album_by_year",
                "description": "Get the Grammy Album of the Year for a specific year.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "year": {
                            "type": "integer",
                            "description": "The year interested.",
                        }
                    },
                    "required": ["year"],
                },
            },
        }

music_grammy_get_all_awarded_artists = {
            "type": "function",
            "function": {
                "name": "music_grammy_get_all_awarded_artists",
                "description": "Get all artists awarded the Grammy Best New Artist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                    },
                    "required": [],
                },
            },
        }

sports_soccer_get_games_on_date = {
            "type": "function",
            "function": {
                "name": "sports_soccer_get_games_on_date",
                "description": "Get soccer games on a specific date. Result includes game attributes such as date, time, GF: GF: Goals For - the number of goals scored by the team in question during the match, GA: Goals Against - the number of goals conceded by the team during the match, xG: Expected Goals - a statistical measure that estimates the number of goals a team should have scored based on the quality of chances they created, xGA: Expected Goals Against - a measure estimating the number of goals a team should have conceded based on the quality of chances allowed to the opponent, Poss: Possession - the percentage of the match time during which the team had possession of the ball.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "The date interested. It can be a time period. Must be in format of %Y-%m-%d, %Y-%m or %Y, e.g. 2024-03-01, 2024-03, 2024",
                        },
                        "team_name": {
                            "type": "string",
                            "description": "The team interested.",
                        }
                    },
                    "required": ["date", "team_name"],
                },
            },
        }

sports_nba_get_games_on_date = {
            "type": "function",
            "function": {
                "name": "sports_nba_get_games_on_date",
                "description": "Get NBA games on a specific date. Result includes game attributes such as team_name_home: The full name of the home team, wl_home: The outcome of the game for the home team, pts_home: The total number of points scored by the home team.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "The date interested. It can be a time period",
                        },
                        "team_name": {
                            "type": "string",
                            "description": "The team interested.",
                        }
                    },
                    "required": ["date", "team_name"],
                },
            },
        }

sports_nba_get_play_by_play_data_by_game_ids = {
            "type": "function",
            "function": {
                "name": "sports_nba_get_play_by_play_data_by_game_ids",
                "description": "Get NBA play by play data for a set of game ids. Result includes play-by-play event time, description, player etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "game_ids": {
                            "type": "list",
                            "description": "List of game ids.",
                        }
                    },
                    "required": ["game_ids"],
                },
            },
        }

open_tools = [open_search_entity_by_name, open_get_entity]
movie_tools =[movie_get_person_info, movie_get_movie_info, movie_get_year_info, movie_get_person_info_by_id, movie_get_movie_info_by_id]
finance_tools_A = [finance_get_company_name, finance_get_price_history, finance_get_detailed_price_history, finance_get_dividends_history, finance_get_market_capitalization, finance_get_eps, finance_get_pe_ratio, finance_get_info]
finance_tools_B = [finance_get_company_name, finance_get_ticker_by_name, finance_get_price_history, finance_get_detailed_price_history, finance_get_dividends_history, finance_get_market_capitalization, finance_get_eps, finance_get_pe_ratio, finance_get_info]
music_tools = [
         music_search_artist_entity_by_name, music_search_song_entity_by_name, music_get_billboard_attributes, music_get_billboard_rank_date,
         music_grammy_get_best_album_by_year, music_grammy_get_all_awarded_artists, music_grammy_get_award_count_by_song,
         music_grammy_get_best_song_by_year, music_grammy_get_award_date_by_artist, music_grammy_get_best_album_by_year,
         music_grammy_get_all_awarded_artists, music_get_artist_birth_place, music_get_artist_birth_date,
         music_get_members, music_get_lifespan, music_get_song_author, music_get_song_release_country, music_get_song_release_date,
         music_get_artist_all_works, music_grammy_get_award_count_by_artist]
sports_tools = [sports_soccer_get_games_on_date, sports_nba_get_games_on_date, sports_nba_get_play_by_play_data_by_game_ids]

def function_intro(tools):
    results = ""
    for tool in tools:
        results += (
            f"Use the function '{tool['function']['name']}' to '{tool['function']['description']}':\n"
            f"{json.dumps(tool)}\n"
        )
    return results