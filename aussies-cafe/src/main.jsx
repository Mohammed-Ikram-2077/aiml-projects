import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ArrowDown, ArrowRight, ArrowUpRight, Instagram, MapPin, Menu as MenuIcon, X } from 'lucide-react';
import './styles.css';

const menuItems = [
  { name: 'Espresso', desc: 'A short, rich shot of house espresso', price: '$3.25', tag: 'COFFEE', image: 'photo-1510707577719-ae7c14805e3a' },
  { name: 'Americano', desc: 'Espresso topped with hot water', price: '$3.75', tag: 'COFFEE', image: 'photo-1509042239860-f550ce710b93' },
  { name: 'Cappuccino', desc: 'Espresso, steamed milk and soft foam', price: '$4.50', tag: 'CLASSIC', image: 'photo-1572442388796-11668a67e53d' },
  { name: 'Latte', desc: 'Smooth espresso with steamed milk', price: '$4.50', tag: 'FAVOURITE', image: 'photo-1461023058943-07fcbe16d735' },
  { name: 'Flat white', desc: 'Aussie classic, silky and balanced', price: '$4.50', tag: 'HOUSE PICK', image: 'photo-1442512595331-e89e73853f31' },
  { name: 'Mocha', desc: 'Espresso, chocolate and steamed milk', price: '$5.00', tag: 'COFFEE', image: 'photo-1570968915860-54d5c301fa9f' },
  { name: 'Iced coffee', desc: 'Chilled espresso over ice and milk', price: '$5.00', tag: 'ICED', image: 'photo-1517701604599-bb29b565090c' },
  { name: 'Cold brew', desc: 'Slow-steeped, smooth and refreshing', price: '$5.25', tag: 'SLOW BREW', image: 'photo-1517701550927-30cf4ba1dba5' },
  { name: 'Chai latte', desc: 'Spiced chai with creamy steamed milk', price: '$4.75', tag: 'TEA', image: 'photo-1571934811356-5cc061b6821f' },
  { name: 'Iced matcha', desc: 'Ceremonial matcha, oat milk and ice', price: '$5.50', tag: 'FRESH', image: 'photo-1515823064-d6e0c04616a7' },
  { name: 'Hot chocolate', desc: 'Rich cocoa finished with soft milk foam', price: '$4.75', tag: 'COZY', image: 'photo-1542990253-0b8be4a2b6ce' },
  { name: 'Almond croissant', desc: 'Twice-baked, golden, flaky', price: '$4.25', tag: 'BAKED DAILY', image: 'photo-1555507036-ab1f4038808a' },
  { name: 'Blueberry muffin', desc: 'Soft-baked with juicy blueberries', price: '$3.75', tag: 'BAKED DAILY', image: 'photo-1607958996333-41aef7caefaa' },
  { name: 'Avocado toast', desc: 'Smashed avocado, sourdough and herbs', price: '$9.50', tag: 'BRUNCH', image: 'photo-1525351484163-7529414344d8' },
  { name: 'Pancakes', desc: 'Fluffy stack with berries and maple syrup', price: '$11.00', tag: 'BRUNCH', image: 'photo-1528207776546-365bb710ee93' },
  { name: 'Grilled cheese', desc: 'Golden sourdough with melted cheese', price: '$8.50', tag: 'ALL DAY', image: 'photo-1528735602780-2552fd46c7af' },
  { name: 'Chicken sandwich', desc: 'Grilled chicken, greens and house sauce', price: '$10.50', tag: 'ALL DAY', image: 'photo-1606755962773-d324e0a13086' },
  { name: 'Garden salad', desc: 'Crunchy greens, seasonal veg and dressing', price: '$9.00', tag: 'FRESH', image: 'photo-1512621776951-a57141f2eefd' },
];
const photo = (id, width=1000) => `https://images.unsplash.com/${id}?auto=format&fit=crop&w=${width}&q=85`;

function App() {
  const [scroll, setScroll] = useState(0);
  const [menuOpen, setMenuOpen] = useState(false);
  useEffect(() => {
    const onScroll = () => setScroll(window.scrollY);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);
  const heroTilt = Math.min(scroll / 34, 11);
  const parallax = Math.min(scroll * .22, 110);
  const menuProgress = Math.max(0, Math.min((scroll - 950) / 500, 1));
  return <>
    <header className="nav">
      <a className="wordmark" href="#home" aria-label="Aussies home">aussies<span>®</span></a>
      <nav className={menuOpen ? 'nav-links open' : 'nav-links'}>
        <a onClick={() => setMenuOpen(false)} href="#menu">The menu</a><a onClick={() => setMenuOpen(false)} href="#story">Our story</a><a onClick={() => setMenuOpen(false)} href="#visit">Find us</a>
      </nav>
      <a className="nav-cta" href="#visit">Come on in <ArrowUpRight size={15}/></a>
      <button className="menu-toggle" onClick={() => setMenuOpen(!menuOpen)} aria-label="Toggle navigation">{menuOpen ? <X/> : <MenuIcon/>}</button>
    </header>

    <main>
      <section className="hero" id="home">
        <div className="hero-copy">
          <p className="eyebrow"><span className="live-dot"/> GOOD COFFEE. GOOD COMPANY.</p>
          <h1>Your daily<br/><em>little</em> escape.</h1>
          <p className="hero-sub">A neighborhood kind of place, with a little bit of Aussie sunshine in every cup.</p>
          <a href="#menu" className="pill-button">Explore the menu <ArrowRight size={16}/></a>
          <div className="scroll-note"><span className="scroll-line"/><span>SCROLL TO SOAK IT IN</span><ArrowDown size={14}/></div>
        </div>
        <div className="hero-stage" style={{ perspective: '1100px' }}>
          <div className="hero-photo-wrap" style={{ transform: `translateY(${-parallax}px) rotateX(${heroTilt}deg)` }}>
            <img className="hero-photo" src={photo('photo-1445116572660-236099ec97a0', 1600)} alt="Sunlit modern café interior" />
            <div className="photo-shade"/>
            <div className="hero-caption"><span>YOUR NEIGHBORHOOD<br/>COFFEE SPOT</span><span className="caption-mark">✳</span></div>
          </div>
          <div className="floating-sticker" style={{ transform: `translateY(${parallax * .45}px) rotate(${8 - heroTilt}deg)` }}>BREWED<br/>WITH LOVE<span>✳</span></div>
          <div className="hero-index">01 <span/> 04</div>
        </div>
        <div className="hero-bottom"><span>MADE FOR SLOW MORNINGS</span><span>EST. EVERY DAY&nbsp; · &nbsp;YOUR PLACE</span></div>
      </section>

      <section className="intro section-pad" id="story">
        <div className="section-label"><span>01 / THE FEELING</span><span>MORE THAN A COFFEE RUN</span></div>
        <div className="intro-grid"><p className="intro-overline">THE AUSSIES WAY</p><div><h2>A good cup is<br/>just the <em>beginning.</em></h2><p className="intro-text">We believe the best parts of your day happen in between. One more sip, one more story, one more reason to stay a little longer. Pull up a chair — we saved you a spot.</p><a className="text-link" href="#visit">A little about us <ArrowRight size={16}/></a></div></div>
      </section>

      <section className="menu-section section-pad" id="menu">
        <div className="section-label"><span>02 / GOOD THINGS</span><span>MADE FRESH. ALWAYS.</span></div>
        <div className="menu-heading"><div><p className="eyebrow">A LITTLE SOMETHING FOR EVERYONE</p><h2>The good<br/><em>stuff.</em></h2></div><a className="circle-link" href="#visit" aria-label="Visit us"><ArrowDown size={21}/></a></div>
        <div className="menu-grid">{menuItems.map((item, i) => <article className="menu-card" key={item.name} style={{ '--depth': `${(i % 3) * 16}px`, '--scroll-depth': `${menuProgress * 18}px`, '--tilt-x': `${(1 - menuProgress) * (i % 2 ? 3 : -3)}deg`, '--tilt-y': `${(1 - menuProgress) * (i % 2 ? -4 : 4)}deg` }}>
          <div className="menu-image"><img src={photo(item.image, 800)} alt={item.name}/><span className="item-tag">{item.tag}</span><span className="item-number">0{i+1}</span></div>
          <div className="menu-item-head"><h3>{item.name}</h3><span>{item.price}</span></div><p>{item.desc}</p>
        </article>)}</div>
        <p className="menu-footnote">From the first coffee to a long lunch — there's something for every kind of day.</p>
      </section>

      <section className="feature">
        <div className="feature-image"><img src={photo('photo-1495474472287-4d71bcdd2085', 1600)} alt="Freshly brewed coffee on the counter" style={{ transform: `translateY(${Math.min(scroll * .08, 30)}px)` }}/><span className="feature-stamp">A GOOD<br/>DAY STARTS<br/>RIGHT HERE</span></div>
        <div className="feature-copy"><span className="eyebrow">A PLACE TO LAND</span><h2>Stay for<br/>another <em>round.</em></h2><p>For the first coffee, the long lunch, and the “just five more minutes”. Our door is open, the coffee’s on, and there’s always room for one more.</p><a className="pill-button light" href="#visit">Make yourself at home <ArrowRight size={16}/></a></div>
      </section>

      <section className="visit section-pad" id="visit">
        <div className="section-label"><span>03 / DROP BY</span><span>WE'LL PUT THE KETTLE ON</span></div>
        <div className="visit-grid"><div><p className="eyebrow">YOUR TABLE'S WAITING</p><h2>See you<br/><em>soon?</em></h2><a className="pill-button" href="https://maps.google.com/?q=cafes+near+me" target="_blank" rel="noreferrer">Get directions <ArrowUpRight size={16}/></a></div>
          <div className="visit-details"><div className="detail-block"><MapPin size={19}/><div><h3>Find our little corner</h3><p>Your neighbourhood, your local spot.<br/>Pop by and say hello.</p></div></div><div className="detail-block"><span className="clock-icon">◷</span><div><h3>Opening hours</h3><p>Monday – Friday&nbsp;&nbsp; 7am – 4pm<br/>Saturday – Sunday&nbsp;&nbsp; 8am – 3pm</p></div></div><a className="social-link" href="https://instagram.com" target="_blank" rel="noreferrer"><Instagram size={17}/> Follow along <ArrowUpRight size={14}/></a></div></div>
      </section>
    </main>
    <footer className="footer"><a className="wordmark" href="#home">aussies<span>®</span></a><span>GOOD COFFEE, GOOD COMPANY.</span><span>© AUSSIES 2025</span><a href="#home" className="back-top">BACK TO TOP ↑</a></footer>
  </>;
}

createRoot(document.getElementById('root')).render(<App/>);
