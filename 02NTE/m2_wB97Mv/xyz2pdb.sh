for i in huangm06xyz/* 
do 
   molecule=${i%.*}
   molecule=${molecule#*/}
   pdbfile=$molecule'.pdb'
   ase convert $i mol2m06pdb/$pdbfile
   rm $i
   echo $pdbfile
done
