for i in molxyz/* 
do 
   molecule=${i%.*}
   molecule=${molecule#*/}
   pdbfile=$molecule'.pdb'
   ase convert $i molpdb/$pdbfile
   rm $i
   echo $pdbfile
done
