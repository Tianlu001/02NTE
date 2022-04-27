for i in molpdb/* 
do 
   molecule=${i%.*}
   molecule=${molecule#*/}
   sdffile=$molecule'.sdf'
   python read_pdb.py $i molsdf/$sdffile
   echo $sdffile
done
