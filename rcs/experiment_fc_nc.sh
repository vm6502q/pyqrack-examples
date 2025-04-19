# Suggested value for SDRP:
export QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.02

for w in {2..50} ; do QRACK_QTENSORNETWORK_THRESHOLD_QB=49 python3 fc_nc.py $w $w; done
