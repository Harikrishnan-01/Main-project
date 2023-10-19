import React from 'react'
import '../App.css'
import bg from './images/bg.png';
import { Link } from 'react-scroll';
const Home = () => {
  return (
    <div className='home-layout' style={{ backgroundImage: `url(${bg})`,width: '100%' }}>

      <div className='overlay'>
        <h3>SPEAK WITH YOUR HANDS....</h3>
        <h1>SIGN LANGUAGE</h1>
        <Link to='features-sec' smooth={true} duration={500} className='btn'>
          GET STARTED...
        </Link>
        
        
      </div>
    </div>
  )
}

export default Home