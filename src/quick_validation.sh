echo test $1/TextZoom/test/easy/ with aster, batch size=$2
python3 main.py --arch tsrn --batch_size=$2 --test --test_data_dir=$1/TextZoom/test/easy/ --resume=../ckpt/with_test/model_best.pth --STN --mask --gradient --nonlocal_type=transformer --srb 4 --save_dir=../ckpt/with_test/result/easy --conv_num=2 --config_path ../ckpt/with_test  --hd_u 64  | tee -a  ../ckpt/with_test/result.txt

echo test $1/TextZoom/test/medium/  with aster
python3 main.py --arch tsrn --batch_size=$2 --test --test_data_dir=$1/TextZoom/test/medium/ --resume=../ckpt/with_test/model_best.pth --STN --mask --gradient --nonlocal_type=transformer --srb 4 --save_dir=../ckpt/with_test/result/medium --conv_num=2 --config_path ../ckpt/with_test --hd_u 64  | tee -a  ../ckpt/with_test/result.txt

echo test $1/TextZoom/test/hard/  with aster
python3 main.py --arch tsrn --batch_size=$2 --test --test_data_dir=$1/TextZoom/test/hard/ --resume=../ckpt/with_test/model_best.pth --STN --mask --gradient --nonlocal_type=transformer --srb 4 --save_dir=../ckpt/with_test/result/hard --conv_num=2 --config_path ../ckpt/with_test  --hd_u 64 | tee -a  ../ckpt/with_test/result.txt


echo test $1/TextZoom/test/easy/ with moran
python3 main.py --arch tsrn --rec moran --batch_size=$2 --test --test_data_dir=$1/TextZoom/test/easy/ --resume=../ckpt/with_test/model_best.pth --STN --mask --gradient --nonlocal_type=transformer --srb 4 --save_dir=../ckpt/with_test/result/easy --conv_num=2 --config_path ../ckpt/with_test  --hd_u 64  | tee -a  ../ckpt/with_test/result.txt

echo test $1/TextZoom/test/medium/  with moran
python3 main.py --arch tsrn --rec moran --batch_size=$2 --test --test_data_dir=$1/TextZoom/test/medium/ --resume=../ckpt/with_test/model_best.pth --STN --mask --gradient --nonlocal_type=transformer --srb 4 --save_dir=../ckpt/with_test/result/medium --conv_num=2 --config_path ../ckpt/with_test --hd_u 64  | tee -a  ../ckpt/with_test/result.txt

echo test $1/TextZoom/test/hard/  with moran
python3 main.py --arch tsrn --rec moran --batch_size=$2 --test --test_data_dir=$1/TextZoom/test/hard/ --resume=../ckpt/with_test/model_best.pth --STN --mask --gradient --nonlocal_type=transformer --srb 4 --save_dir=../ckpt/with_test/result/hard --conv_num=2 --config_path ../ckpt/with_test  --hd_u 64 | tee -a  ../ckpt/with_test/result.txt


echo test $1/TextZoom/test/easy/ with crnn
python3 main.py --arch tsrn --rec crnn --batch_size=$2 --test --test_data_dir=$1/TextZoom/test/easy/ --resume=../ckpt/with_test/model_best.pth --STN --mask --gradient --nonlocal_type=transformer --srb 4 --save_dir=../ckpt/with_test/result/easy --conv_num=2 --config_path ../ckpt/with_test  --hd_u 64  | tee -a  ../ckpt/with_test/result.txt

echo test $1/TextZoom/test/medium/  with crnn
python3 main.py --arch tsrn --rec crnn --batch_size=$2 --test --test_data_dir=$1/TextZoom/test/medium/ --resume=../ckpt/with_test/model_best.pth --STN --mask --gradient --nonlocal_type=transformer --srb 4 --save_dir=../ckpt/with_test/result/medium --conv_num=2 --config_path ../ckpt/with_test --hd_u 64  | tee -a  ../ckpt/with_test/result.txt

echo test $1/TextZoom/test/hard/  with crnn
python3 main.py --arch tsrn --rec crnn --batch_size=$2 --test --test_data_dir=$1/TextZoom/test/hard/ --resume=../ckpt/with_test/model_best.pth --STN --mask --gradient --nonlocal_type=transformer --srb 4 --save_dir=../ckpt/with_test/result/hard --conv_num=2 --config_path ../ckpt/with_test  --hd_u 64 | tee -a  ../ckpt/with_test/result.txt



echo test $1/low_res_all/ with aster
python3 main.py --rec aster --arch tsrn --batch_size=$2 --test --test_data_dir=$1/low_res_all/ --resume=../ckpt/with_test/model_best.pth --STN --mask --gradient --nonlocal_type=transformer --srb 4 --save_dir=../ckpt/with_test/result/test/res_low_all --conv_num=2 --config_path ../ckpt/with_test --hd_u 64  | tee -a  ../ckpt/with_test/result.txt

echo test $1/low_res_all/ with moran
python3 main.py --rec moran --arch tsrn --batch_size=$2 --test --test_data_dir=$1/low_res_all/ --resume=../ckpt/with_test/model_best.pth --STN --mask --gradient --nonlocal_type=transformer --srb 4 --save_dir=../ckpt/with_test/result/test/res_low_all --conv_num=2 --config_path ../ckpt/with_test --hd_u 64  | tee -a  ../ckpt/with_test/result.txt

echo test $1/low_res_all/ with crnn
python3 main.py --rec crnn  --arch tsrn --batch_size=$2 --test --test_data_dir=$1/low_res_all/ --resume=../ckpt/with_test/model_best.pth --STN --mask --gradient --nonlocal_type=transformer --srb 4 --save_dir=../ckpt/with_test/result/test/res_low_all --conv_num=2 --config_path ../ckpt/with_test --hd_u 64  | tee -a  ../ckpt/with_test/result.txt
