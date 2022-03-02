source ${1:-../sparc_setup/sparc_env.sh}
python h2o_reaction_energies.py 2>&1 | tee h2o_reaction_energies.log
