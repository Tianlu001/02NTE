rm error.txt
for(( i =1; i<=20; i++))
do
    python trainntest.py >> error.txt
   #python mass_kmatrix.py >> error.txt
done

python mean.py

