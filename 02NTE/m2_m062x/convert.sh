for i in huangm06xyz/* 
do 
   molecule=${i%.*}
   molecule=${molecule#*/}
   sdffile=$molecule'.sdf'
   xyz2mol.py $i -o sdf > molsdf/$sdffile
   echo $sdffile
done
