import compiam

ftanet_carnatic = compiam.load_model("melody:ftanet-carnatic")
ftanet_pitch_track = ftanet_carnatic.predict('data/audio/raksha_bettare.wav',hop_size=30)
write_pitch_track(ftanet_pitch_track, 'data/pitch_tracks/raw/raksha_bettare_new.tsv', sep='\t')