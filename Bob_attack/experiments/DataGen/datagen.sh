if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
else
    echo "conda not found in PATH"
    exit 1
fi

source $CONDA_BASE/bin/activate bob

# Generate Bob's (helper, image) training pairs for the facenet PIPE-FE schemes, reproducing
# the PIPE datagen helpers bit-for-bit. Output ->
#   ./databases/celeba/facenet/<scheme>[_centered]_none/{train,test}/{images,embeddings}

# unprotected
python GenDataset.py --same_r none --prot_scheme unprotected --mode train &
python GenDataset.py --same_r none --prot_scheme unprotected --mode test
echo "generated unprotected facenet dataset"

# facialfe_e8 (E8 secure sketch)
python GenDataset.py --same_r none --prot_scheme facialfe_e8 --mode train &
python GenDataset.py --same_r none --prot_scheme facialfe_e8 --mode test
echo "generated facialfe_e8 facenet dataset"

# facialfe_leech (centered Leech secure sketch)
python GenDataset.py --same_r none --prot_scheme facialfe_leech --centered --mode train &
python GenDataset.py --same_r none --prot_scheme facialfe_leech --centered --mode test
echo "generated facialfe_leech facenet dataset (centered)"

# facialfe_leech_randomised (centered Leech sketch, random per-id codeword)
python GenDataset.py --same_r none --prot_scheme facialfe_leech_randomised --centered --mode train &
python GenDataset.py --same_r none --prot_scheme facialfe_leech_randomised --centered --mode test
echo "generated facialfe_leech_randomised facenet dataset (centered)"

# ourfe 
python GenDataset.py --same_r none --prot_scheme ourfe --mode train &
python GenDataset.py --same_r none --prot_scheme ourfe --mode test
echo "generated ourfe facenet dataset"
