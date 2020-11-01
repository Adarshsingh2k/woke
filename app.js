var rand=Math.floor((Math.random() * 100000) + 1);

function otpone(){

    alert("OTP sent to Phone Number");
    console.log(rand);
};

document.getElementById("verify").addEventListener("click",function(){
    
    alert("OTP Verification Sucessfull");
});