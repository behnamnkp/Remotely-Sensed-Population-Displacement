

Input: (G, M, )
day = 0
initialize_environment()
initialize_population()
initialize_micromobility()
for timestep in time_range(start, end, hour):
    identify_day_time()
    relocate_population()
    starting_micromobility_trips()
    assign_riders()
    if timestep =='12 AM':
        intervention(hand sanitizing, micromobility disinfection)
        micromobility_to_human_transmission()
        human_to_micromobility_transmission()
        control_populaiton_transmission()
        update_infection_contamination()
        recovery()
